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

docs:
	$(SPHINXBUILD) $(SPHINXOPTS) $(DOCS_SRC) $(DOCS_OUT)

docs-clean:
	rm -rf $(DOCS_OUT)

docs-serve:
	$(PYTHON) -m http.server 8000 -d $(DOCS_OUT)

init:
	$(PRECOMMIT) install

lint:
	$(PRECOMMIT) run --all-files --show-diff-on-failure $(ARGS)

test:
	$(COVERAGE) run -p -m pytest -v

cov:
	$(COVERAGE) combine
	$(COVERAGE) report
	$(COVERAGE) xml

type:
	$(MYPY) src tests docs/conf.py

xdoc:
	$(XDOCTEST) clophfit all

all: test type xdoc cov


# git cliff --bump --unreleased --prepend CHANGELOG.md
ch:
	set -euo pipefail; \
	git cliff --bump --unreleased -o RELEASE.md; \
	$(UV) run python scripts/update_changelog.py --raw RELEASE.md --changelog CHANGELOG.md; \
	rm -f RELEASE.md; \
	echo "CHANGELOG.md updated."

bump:
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
clean:
	rm -rf ./build .coverage ./__pycache__ ./.mypy_cache ./.pytest_cache ./docs/_build ./tests/__pycache__ ./dist ./src/clophfit/__pycache__
