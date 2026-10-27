#!/usr/bin/env bash
# Restore Ellf + Codex on the dacy job after a restart.
#
# Keep this script in your persistent UCloud folder. Everything it needs is
# stored in a ".ellf-dacy" folder next to it, so nothing depends on the job's
# own /work folder (which is wiped when the job restarts).
#
# Usage (bash, from any directory; quote the path because of the "#"):
#   source "/work/UCLOUD#USERNAME/setup-ellf-ucloud.sh"
#
# "source" (not "bash") matters: it sets PATH, the venv and env vars in your shell.
#
# The cluster is reached over its public hostname, so no SSH tunnel or
# private-network link to the VM is needed. If the VM's public IP ever
# changes, update ELLF_HOST below, and on the cluster side rerun the Helm
# upgrade (cluster.fqdn) and `ellf clusters update ellf-cluster --address ...`.
#
# The first run installs ellf-cli and Codex automatically and tells you about
# the few one-time manual steps (CA certificate, logins).
#
# Start Codex afterwards with:  codex-safe
# It runs as a restricted "codex" user that can only write in $CODEX_WORK
# (set below), has no sudo, and can't read your home folder or drive.

# Public address of the Ellf cluster
ELLF_HOST="https://130-225-164-142.sslip.io"

# Folder the restricted "codex" user may work in (it can't read anything else of yours)
CODEX_WORK="/work/UCLOUD#USERNAME/DIR_FOR_AGENT_TO_WORK_IN"

_ellf_setup() {
  # --- Locations ---------------------------------------------------------
  local script_dir store
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  store="$script_dir/.ellf-dacy"            # real, persistent storage
  local D="$HOME/.ellf-dacy"                # stable link to it (no "#" or "ø" in paths)
  mkdir -p "$store"
  chmod 700 "$store"
  ln -sfn "$store" "$D"

  local CA_FILE="$D/ca.crt"
  local VENV="$D/venv"
  local CODEX_BIN="$D/codex-cli/bin"
  local CODEX_DIR="$D/codex"
  local STATE_DIR="$D/state"
  mkdir -p "$CODEX_DIR/skills" "$STATE_DIR"

  # --- 1. One-time prerequisites ------------------------------------------
  echo "== 1/7 One-time prerequisites"
  if [ ! -f "$CA_FILE" ]; then
    echo "  ✘ CA certificate missing. On the VM run:  base64 -w0 /home/ucloud/.prodigy-teams/ca.crt; echo"
    echo "    then here:  echo 'PASTED_LINE' | base64 -d > ~/.ellf-dacy/ca.crt"
    echo "  Fix this, then source this script again."
    return 1
  fi
  echo "  ✔ CA certificate present"

  # --- 2. CA trust --------------------------------------------------------
  echo "== 2/7 CA certificate"
  sudo cp "$CA_FILE" /usr/local/share/ca-certificates/prodigy-teams-ca.crt
  sudo update-ca-certificates >/dev/null && echo "  ✔ CA trusted"

  # --- 3. Login state -----------------------------------------------------
  echo "== 3/7 Login state"
  _ellf_persist "$HOME/.config/ellf" "$STATE_DIR/config-ellf"
  _ellf_persist "$HOME/.claude.json" "$STATE_DIR/claude.json"
  _ellf_persist "$HOME/.claude"      "$STATE_DIR/claude-dir"
  echo "  ✔ login files linked from storage"

  # --- 4. ellf venv -------------------------------------------------------
  echo "== 4/7 ellf-cli"
  if ! "$VENV/bin/python" -c 'import sys' >/dev/null 2>&1; then
    echo "  ℹ installing ellf-cli (first run, ~1 min)..."
    rm -rf "$VENV"
    python3 -m venv "$VENV" && "$VENV/bin/pip" install -q ellf-cli \
      || { echo "  ✘ ellf-cli install failed"; return 1; }
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  export ELLF_BROKER_HOST="$ELLF_HOST"
  echo "  ✔ ellf $(ellf --version 2>/dev/null) active, cluster at $ELLF_BROKER_HOST"

  # --- 5. Codex -----------------------------------------------------------
  echo "== 5/7 Codex"
  if [ ! -x "$CODEX_BIN/codex" ]; then
    echo "  ℹ installing Codex (first run)..."
    npm install -g -s --prefix "$D/codex-cli" @openai/codex \
      || { echo "  ✘ Codex install failed"; return 1; }
  fi
  case ":$PATH:" in
    *":$CODEX_BIN:"*) ;;
    *) export PATH="$CODEX_BIN:$PATH" ;;
  esac
  export CODEX_HOME="$CODEX_DIR"
  # Restore the UCloud inference token from persistent, private storage.
  # An already-exported token takes precedence over the saved one.
  local UCLOUD_TOKEN_FILE="$D/ucloud-inference-token"
  if [ -f "$UCLOUD_TOKEN_FILE" ]; then
    chmod 600 "$UCLOUD_TOKEN_FILE" \
      || { echo "  ✘ could not protect UCloud key file"; return 1; }
    if [ -z "${UCLOUD_INFERENCE_TOK:-}" ]; then
      export UCLOUD_INFERENCE_TOK="$(cat "$UCLOUD_TOKEN_FILE")"
    fi
  fi
  # MCP server entry (added once)
  if ! grep -q '^\[mcp_servers.ellf-kb\]' "$CODEX_DIR/config.toml" 2>/dev/null; then
    cat >> "$CODEX_DIR/config.toml" <<'EOF'

[mcp_servers.ellf-kb]
url = "https://app.ellf.ai/ellf-core/mcp"
bearer_token_env_var = "ELLF_KB_TOKEN"
EOF
  fi
  # Ellf skills, copied from the Claude Code install that `ellf login --claude` makes
  if ls "$HOME"/.claude/skills/ellf-* >/dev/null 2>&1; then
    cp -r "$HOME"/.claude/skills/ellf-* "$CODEX_DIR/skills/"
  fi
  # MCP token (never printed)
  ELLF_KB_TOKEN=$(python3 - <<'PY' 2>/dev/null
import json, os
cfg = json.load(open(os.path.expanduser("~/.claude.json")))
print(cfg["mcpServers"]["ellf-kb"]["headers"]["Authorization"].split(" ", 1)[1])
PY
)
  local todo=0
  if [ -n "$ELLF_KB_TOKEN" ]; then
    export ELLF_KB_TOKEN
  else
    unset ELLF_KB_TOKEN
    echo "  ✘ Not logged in to Ellf. Run:  ellf login --claude --no-browser   then source this again."
    todo=1
  fi
  # UCloud authenticates through UCLOUD_INFERENCE_TOK; an OpenAI login is optional.
  if [ -z "${UCLOUD_INFERENCE_TOK:-}" ]; then
    echo "  ✘ UCloud key missing. Export UCLOUD_INFERENCE_TOK or save it in:"
    echo "      $UCLOUD_TOKEN_FILE (mode 600)"
    todo=1
  fi
  [ "$todo" = 0 ] && echo "  ✔ $(codex --version) ready, Ellf + UCloud tokens loaded"

  # --- 6. Cluster check ---------------------------------------------------
  echo "== 6/7 Cluster check"
  if [ "$todo" = 1 ]; then
    echo "  ℹ skipped until the logins above are done"
    return 1
  fi
  if ! ellf clusters check; then
    echo "  ✘ Check failed. Is the VM running with its public IP attached?"
    echo "    If you're logged out, run 'ellf login --claude --no-browser', then source this again."
    return 1
  fi

  # --- 7. Restricted Codex user -------------------------------------------
  # Codex runs as its own Linux user "codex" (no sudo) that can write only in
  # $CODEX_WORK. Codex's built-in sandbox can't run in this container, so the
  # separate user is the sandbox instead.
  echo "== 7/7 Restricted Codex user"
  local CU="codex" CG="codexwork" CID=5050
  local CH="/home/$CU/.codex"

  # 7a. User and group (fixed IDs, so file ownership stays stable across restarts)
  getent group "$CG" >/dev/null || sudo groupadd -g "$CID" "$CG" \
    || { echo "  ✘ couldn't create group $CG"; return 1; }
  id "$CU" >/dev/null 2>&1 || sudo useradd -m -u "$CID" -g "$CG" -s /bin/bash "$CU" \
    || { echo "  ✘ couldn't create user $CU"; return 1; }
  sudo chmod 700 "/home/$CU"
  sudo usermod -aG "$CG" "$USER"        # lets you edit files Codex creates (new shells)
  if sudo -l -U "$CU" 2>/dev/null | grep -q 'may run'; then
    echo "  ✘ user $CU has sudo rights. Stopping; this must not happen."
    return 1
  fi

  # 7b. Share only the work folder with the group.
  # Let the group pass through every folder from /work down to CODEX_WORK
  # (traverse only: x without r, so it can't list or read them; the other
  # folders in /work keep their own ucloud-only permissions)
  local d="$(dirname "$CODEX_WORK")"
  while :; do
    sudo chgrp "$CG" "$d" && sudo chmod g=x "$d" \
      || { echo "  ✘ couldn't set traverse on $d"; return 1; }
    [ "$d" = "/work" ] && break
    d="$(dirname "$d")"
  done
  sudo find "$CODEX_WORK" ! -group "$CG" -exec chgrp -h "$CG" {} +
  sudo find "$CODEX_WORK" -type d ! -perm -2070 -exec chmod g+rwxs {} +
  sudo find "$CODEX_WORK" -type f ! -perm -g+rw -exec chmod g+rw {} +

  # 7c. Tools in /opt (root-owned, so Codex can use but not change them)
  if [ ! -x /opt/codex-cli/bin/codex ]; then
    sudo cp -a "$D/codex-cli" /opt/codex-cli && sudo chown -R root:root /opt/codex-cli
  fi
  if ! /opt/ellf-venv/bin/python -c 'import sys' >/dev/null 2>&1; then
    echo "  ℹ installing ellf-cli for the codex user..."
    sudo rm -rf /opt/ellf-venv
    sudo python3 -m venv /opt/ellf-venv \
      && sudo -H /opt/ellf-venv/bin/pip install -q --no-cache-dir ellf-cli \
      || { echo "  ✘ ellf-cli install for codex user failed"; return 1; }
  fi

  # 7d. Codex user's own config: UCloud provider, Ellf login, skills, MCP server
  sudo install -d -o "$CU" -g "$CG" -m 700 "$CH" "$CH/skills" "/home/$CU/.config/ellf"
  # Optional: keep OpenAI login for manually switching back to that provider.
  if [ -f "$CODEX_DIR/auth.json" ]; then
    sudo cp "$CODEX_DIR/auth.json" "$CH/auth.json"
    sudo chmod 600 "$CH/auth.json"
  fi
  sudo cp -r "$CODEX_DIR/skills/." "$CH/skills/"
  sudo cp -rL "$HOME/.config/ellf/." "/home/$CU/.config/ellf/"
  sudo tee "$CH/config.toml" >/dev/null <<'EOF'
# Written by setup-ellf-ucloud.sh on every run; edit the script, not this file.
# Codex's own sandbox can't run in this container. The "codex" Linux user
# (write access only to the work folder, no sudo) is the sandbox instead.
model = "zai-org/GLM-5.3"
model_provider = "ucloud"
sandbox_mode = "danger-full-access"
# Codex asks when it judges a command needs approval ("untrusted" is no longer supported).
approval_policy = "on-request"

[mcp_servers.ellf-kb]
url = "https://app.ellf.ai/ellf-core/mcp"
bearer_token_env_var = "ELLF_KB_TOKEN"

[model_providers.ucloud]
name = "UCloud"
base_url = "https://ai.cloud.sdu.dk/v1"
env_key = "UCLOUD_INFERENCE_TOK"
wire_api = "responses"
requires_openai_auth = false
EOF
  sudo chown -R "$CU:$CG" "/home/$CU"

  # 7e. Verify the isolation
  local ok=1 p
  sudo -u "$CU" test -w "$CODEX_WORK" \
    || { echo "  ✘ codex user can't write $CODEX_WORK"; ok=0; }
  for p in "$HOME" "$script_dir" "$store"; do
    if sudo -u "$CU" test -r "$p"; then echo "  ✘ codex user can read $p"; ok=0; fi
  done
  for p in /work/*/; do
    [ "${p%/}" = "$CODEX_WORK" ] && continue
    sudo -u "$CU" test -r "$p" && echo "  ⚠ codex user can read ${p%/} (check that this is OK)"
  done
  if ! sudo -u "$CU" -H env \
      ELLF_BROKER_HOST="$ELLF_BROKER_HOST" \
      PATH=/opt/ellf-venv/bin:/usr/bin:/bin \
      ellf clusters check >/dev/null 2>&1; then
    echo "  ✘ codex user can't reach the cluster with ellf"; ok=0
  fi
  [ "$ok" = 1 ] || { echo "  Not starting Codex until the items above are fixed."; return 1; }
  echo "  ✔ codex user: writes only $CODEX_WORK, no sudo, can't read your home or drive, can use ellf"
  echo "  ✔ All set. Start Codex with: codex-safe"
}

# Start Codex as the restricted "codex" user, in the work folder.
codex-safe() {
  if [ -z "${UCLOUD_INFERENCE_TOK:-}" ]; then
    echo "UCloud key missing: source setup-ellf-ucloud.sh after loading or saving the key." >&2
    return 1
  fi
  sudo -u codex -H env \
    ELLF_KB_TOKEN="$ELLF_KB_TOKEN" \
    ELLF_BROKER_HOST="$ELLF_HOST" \
    UCLOUD_INFERENCE_TOK="$UCLOUD_INFERENCE_TOK" \
    CODEX_HOME=/home/codex/.codex \
    PATH=/opt/ellf-venv/bin:/opt/codex-cli/bin:/usr/local/bin:/usr/bin:/bin \
    bash -c 'umask 002; cd "$0" && exec codex "$@"' "$CODEX_WORK" "$@"
}

# Move a home-directory file/dir into storage and leave a symlink behind.
# A real (non-link) home copy, e.g. freshly written by a login, replaces the stored one.
_ellf_persist() {
  local home_path="$1" store_path="$2"
  mkdir -p "$(dirname "$home_path")" "$(dirname "$store_path")"
  if [ -e "$home_path" ] && [ ! -L "$home_path" ]; then
    rm -rf "$store_path"
    mv "$home_path" "$store_path"
  fi
  if [ -e "$store_path" ]; then
    ln -sfn "$store_path" "$home_path"
  fi
}

_ellf_setup "$@"
unset -f _ellf_setup _ellf_persist

# Warn if the script was run with `bash` instead of `source`
(return 0 2>/dev/null) || echo "Note: run it with 'source', so PATH and the venv stay set."
