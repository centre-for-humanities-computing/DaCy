install:
	@echo "--- 🚀 Installing project ---"
	uv sync --group dev --group tests --group docs

static-type-check:
	@echo "--- 🔍 Running static type check ---"
	uv run pyright src/

lint:
	@echo "--- 🧹 Running linters ---"
	uv run ruff format .  							# running ruff formatting
	uv run ruff check src/ --fix  					# running ruff linting
	uv run ruff check tests/ --fix
	uv run ruff check docs/conf.py --fix

test:
	@echo "--- 🧪 Running tests ---"
	uv run pytest tests/

pr:
	@echo "--- 🚀 Running PR checks ---"
	make lint
	make static-type-check
	make test
	@echo "Ready to make a PR"

build-docs:
	@echo "--- 📚 Building docs ---"
	@echo "Builds the docs and puts them in the 'site' folder"
	uv run sphinx-build -M html docs/ docs/_build

view-docs:
	@echo "--- 👀 Viewing docs ---"
	@echo "You might need to rebuild the docs first"
	open docs/_build/html/index.html
