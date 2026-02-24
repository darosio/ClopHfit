.. _development:

Development
-----------

Prerequisites:

- `uv <https://docs.astral.sh/uv/>`__ for package management and virtual
  environments.
- `direnv <https://direnv.net/>`__ (recommended) for automatic environment
  activation.
- ``make`` for running development tasks.

Dependencies and their versions are specified in ``pyproject.toml`` (with
dependency groups for dev, lint, tests and docs) and locked in ``uv.lock``.
Lockfile updates are automated via GitHub Actions and Renovate.

Setting up a development environment with direnv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project includes an ``.envrc`` file that automatically creates a virtual
environment (honoring ``.python-version``), activates it and syncs all
dependencies::

   direnv allow
   make init  # install pre-commit hooks

Setting up a development environment manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   uv venv
   uv sync --locked --all-groups --all-extras
   uv run pre-commit install

Available Make targets
~~~~~~~~~~~~~~~~~~~~~~

Run ``make help`` to see all available targets. Key commands::

   make lint       # run pre-commit on all files
   make test       # run pytest with coverage
   make cov        # combine and report coverage
   make type       # run mypy type checking
   make xdoc       # run xdoctest
   make test-all   # run all of the above
   make docs       # build Sphinx documentation
   make docs-serve # serve docs locally

If ``pre-commit`` fails during a push, stage changes, amend the commit, and
push again.

Building docs
~~~~~~~~~~~~~

::

   make docs
   make docs-serve

When needed (e.g. API updates)::

   uv run sphinx-apidoc -f -o docs/api/ src/clophfit/

Bump and releasing
~~~~~~~~~~~~~~~~~~

Version bumping uses ``git-cliff`` for changelog generation and ``uv version``
for version management::

   make bump   # bump version, update changelog, commit and tag
   make ch     # update CHANGELOG.md only

``make bump`` will refuse to run on a dirty working tree. After bumping, push
the commit and tag::

   git push && git push --tags

Release to PyPI is automated via GitHub Actions on tag push.

To keep a clean development history, use branches and PRs::

   gh pr create --fill
   gh pr merge --squash --delete-branch [-t "fix|ci|feat: msg"]

Configuration files
~~~~~~~~~~~~~~~~~~~

Configuration files:

-  pre-commit configured in .pre-commit-config.yaml;
-  ruff (linting and formatting) configured in pyproject.toml;
-  pydoclint configured in pyproject.toml;
-  typos configured in pyproject.toml;
-  coverage configured in pyproject.toml;
-  mypy configured in pyproject.toml;
-  git-cliff configured in cliff.toml;
-  yamlfmt configured in .yamlfmt.yml;
-  taplo (TOML formatting) configured in .taplo.toml;
-  mdformat configured in .mdformat.toml.
