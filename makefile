install:
	@echo "--- 🚀 Installing project ---"
	pip install -e ".[dev, docs, tests]" 

static-type-check:
	@echo "--- 🔍 Running static type check ---"
	pyright src/

lint:
	@echo "--- 🧹 Running linters ---"
	ruff format .  								# running ruff formatting
	ruff src/ --fix  						    # running ruff linting

test:
	@echo "--- 🧪 Running tests ---"
	pytest tests/

pr:
	@echo "--- 🚀 Running PR checks ---"
	make lint
	make static-type-check
	make test
	@echo "Ready to make a PR"

build-docs:
	@echo "--- 📚 Building docs ---"
	@echo "Builds the docs and puts them in the 'site' folder"
	mkdocs build

view-docs:
	@echo "--- 👀 Viewing docs ---"
	mkdocs serve
	
update-from-template:
	@echo "--- 🔄 Updating from template ---"
	@echo "This will update the project from the template, make sure to resolve any .rej files"
	cruft update
	
