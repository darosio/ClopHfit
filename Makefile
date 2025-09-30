SHELL := /bin/bash

# Allow overriding the uv binary if needed (e.g., UV=uvx)
UV ?= uv
UV_RUN := $(UV) run

# Tool shims that always run inside the project env
PYTHON      := $(UV_RUN) python
COVERAGE    := $(UV_RUN) coverage
MYPY        := $(UV_RUN) mypy
PRECOMMIT   := $(UV_RUN) pre-commit
SPHINXBUILD := $(UV_RUN) sphinx-build
XDOCTEST    := $(UV_RUN) python -m xdoctest

# SPHINXOPTS ?= -W
SPHINXOPTS ?=
DOCS_SRC   := docs
DOCS_OUT   := docs/_build
ARGS       ?=

.PHONY: docs docs-clean docs-serve lint test cov type xdoc all ch bump clean


# Documentation
docs:  ## Build docs
	$(SPHINXBUILD) $(SPHINXOPTS) $(DOCS_SRC) $(DOCS_OUT)

docs-clean:  ## Cleans the documentation build directory
	rm -rf $(DOCS_OUT)

docs-serve:  ## Serves the documentation locally
	$(PYTHON) -m http.server 8000 -d $(DOCS_OUT)


# Development setup
init:  ## Installs pre-commit hooks for version control.
	$(PRECOMMIT) install


# Code quality
lint:  ## Lints the codebase using pre-commit.
	$(PRECOMMIT) run --all-files --show-diff-on-failure $(ARGS)


# Testing
test:  ## Runs tests using pytest and coverage
	$(COVERAGE) run -p -m pytest -v

cov:  ## Generates a coverage report in multiple formats (report, xml).
	$(COVERAGE) combine
	$(COVERAGE) report
	$(COVERAGE) xml

type:  ## Checks the type annotations of Python files using mypy.
	$(MYPY) src tests docs/conf.py

xdoc:  ## Runs xdoctest on the project.
	$(XDOCTEST) clophfit all

test-all: test type xdoc cov  ## Runs all tests: testing, type checking, xdoctesting, and generating coverage reports.


# Release management
ch:  ## Bumps the project version number and tags it in Git.
	set -euo pipefail; \
	git cliff --bump --unreleased -o RELEASE.md; \
	$(UV) run python scripts/update_changelog.py --raw RELEASE.md --changelog CHANGELOG.md; \
	rm -f RELEASE.md; \
	echo "CHANGELOG.md updated."
	# git cliff --bump --unreleased --prepend CHANGELOG.md

bump:  ## Bumps the project version number and tags it in Git. It also runs the ch target to create a new release note.
	set -euo pipefail; \
	NEXT_VERSION=$$(git cliff --bumped-version); \
	echo "Bumping to $$NEXT_VERSION"; \
	$(UV) version "$$NEXT_VERSION"; \
	$(UV) lock; \
	$(UV) sync --locked --all-groups; \
	$(MAKE) ch; \
	if ! git diff --quiet; then git add -A && git commit -m "chore: release $$NEXT_VERSION"; else echo "No changes to commit"; fi; \
	git tag -a "$$NEXT_VERSION" -m "Release $$NEXT_VERSION"
	# git push; \
	# git push --tags


# Project cleanup
clean:  ## Project cleanup
	rm -rf ./build .coverage ./__pycache__ ./.mypy_cache ./.pytest_cache ./docs/_build ./tests/__pycache__ ./dist ./src/clophfit/__pycache__


# Help target to show all available commands
help: ## Show this help message.
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
