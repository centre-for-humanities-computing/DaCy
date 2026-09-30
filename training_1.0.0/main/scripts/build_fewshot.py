# Claude
"""
Build a spacy-llm `spacy.NER.v3` few-shot YAML skeleton from a gold .spacy corpus.

Selects sentences from a DocBin under per-label quotas (MISC-heavy by default),
writes their gold entities as positive spans, adds heuristic distractors as
`==NONE==` spans, and leaves every `reason` field blank for you to fill in.

Usage:
    python scripts/build_fewshot.py corpus/dane/train.spacy -o assets/dane_fewshot.yml
    python scripts/build_fewshot.py corpus/dane/train.spacy --quota MISC=8,PER=3,LOC=3,ORG=3
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import spacy
from spacy.tokens import DocBin

NONE_LABEL = "==NONE=="


def parse_quota(s):
    quota = {}
    for part in s.split(","):
        if not part.strip():
            continue
        label, _, n = part.partition("=")
        quota[label.strip()] = int(n)
    return quota


def load_sentences(path, lang):
    """Yield (text, ents) per sentence, where ents are (text, label) in order."""
    nlp = spacy.blank(lang)
    docs = list(DocBin().from_disk(path).get_docs(nlp.vocab))
    for doc in docs:
        units = doc.sents if doc.has_annotation("SENT_START") else [doc[:]]
        for sent in units:
            ents = [e for e in doc.ents if e.start >= sent.start and e.end <= sent.end]
            yield sent, ents


def find_distractors(sent, ents, limit):
    """Capitalized / PROPN tokens outside gold entities — likely false positives."""
    covered = {i for e in ents for i in range(e.start, e.end)}
    out = []
    for tok in sent:
        if tok.i in covered or not tok.text.strip() or tok.is_punct:
            continue
        sentence_initial = tok.i == sent.start
        capitalized = tok.text[0].isupper() and not tok.text.isupper()
        is_propn = tok.pos_ == "PROPN"
        if is_propn or (capitalized and not sentence_initial):
            out.append(tok.text)
    # Preserve order, drop duplicates.
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq[:limit]


def select(candidates, quota, n_empty, seed):
    """Greedy selection: keep a sentence if it fills an unmet label quota."""
    rng = random.Random(seed)
    rng.shuffle(candidates)
    remaining = Counter(quota)
    chosen, empties = [], []
    for sent, ents in candidates:
        labels = Counter(e.label_ for e in ents)
        if not labels:
            if len(empties) < n_empty:
                empties.append((sent, ents))
            continue
        if any(remaining[label] > 0 for label in labels):
            chosen.append((sent, ents))
            remaining.subtract(labels)
    unmet = {k: v for k, v in remaining.items() if v > 0}
    return chosen + empties, unmet


def y(s):
    """Dump a string as a double-quoted YAML scalar (JSON syntax is valid YAML)."""
    return json.dumps(s, ensure_ascii=False)


def render(selected, n_distractors):
    lines = [
        "# Few-shot examples for spacy.NER.v3 — generated skeleton.",
        "# Spans are gold; `==NONE==` spans are heuristic distractors, VERIFY THEM.",
        "# Fill in every `reason` field before using this file.",
        "",
    ]
    for sent, ents in selected:
        text = sent.text
        distractors = find_distractors(sent, ents, n_distractors)
        lines.append(f"- text: {y(text)}")
        if not ents and not distractors:
            lines.append("  spans: []")
            lines.append("")
            continue
        lines.append("  spans:")
        emitted = set()
        for ent in ents:
            if ent.text in emitted:
                continue
            emitted.add(ent.text)
            if text.count(ent.text) > 1:
                lines.append(
                    f"    # NOTE: {y(ent.text)} occurs {text.count(ent.text)}x — "
                    "string matching will tag every occurrence"
                )
            lines.append(f"    - text: {y(ent.text)}")
            lines.append("      is_entity: true")
            lines.append(f"      label: {y(ent.label_)}")
            lines.append(f'      reason: ""  # TODO ({ent.label_})')
        for dist in distractors:
            if dist in emitted:
                continue
            emitted.add(dist)
            lines.append(f"    - text: {y(dist)}")
            lines.append("      is_entity: false")
            lines.append(f"      label: {y(NONE_LABEL)}")
            lines.append('      reason: ""  # TODO (why is this NOT an entity?)')
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus", type=Path, help="Path to a gold .spacy DocBin")
    p.add_argument("-o", "--output", type=Path, default=Path("fewshot.yml"))
    p.add_argument("--lang", default="da")
    p.add_argument(
        "--quota",
        type=parse_quota,
        default="MISC=8,PER=3,LOC=3,ORG=3",
        help="Per-label target counts (default: MISC-heavy)",
    )
    p.add_argument("--n-empty", type=int, default=2, help="Entity-free sentences")
    p.add_argument("--n-distractors", type=int, default=2, help="Max ==NONE== per sentence")
    p.add_argument("--min-tokens", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if isinstance(args.quota, str):
        args.quota = parse_quota(args.quota)

    candidates = [
        (sent, ents)
        for sent, ents in load_sentences(args.corpus, args.lang)
        if args.min_tokens <= len(sent) <= args.max_tokens
    ]
    if not candidates:
        sys.exit("No sentences matched the length filter.")

    selected, unmet = select(candidates, args.quota, args.n_empty, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(selected, args.n_distractors), encoding="utf-8")

    counts = Counter(e.label_ for _, ents in selected for e in ents)
    print(f"Wrote {len(selected)} examples to {args.output}")
    print(f"Label counts: {dict(sorted(counts.items()))}")
    if unmet:
        print(f"WARNING: quota not met for {unmet} — loosen the length filter", file=sys.stderr)


if __name__ == "__main__":
    main()