.. _development:

Development
-----------

You need the following requirements:

-  ``hatch`` for test automation and package dependency managements. If
   you don’t have hatch, you can use ``pipx run hatch`` to run it
   without installing, or ``pipx install hatch``. Dependencies are
   locked thanks to
   `pip-deepfreeze <https://pypi.org/project/pip-deepfreeze/>`__. You
   can run ``hatch env show`` to list available environments and
   scripts.

   ::

        hatch run init  # init repo with pre-commit hooks
        hatch run sync  # sync venv with deepfreeze

        hatch run lint:run
        hatch run tests.py3.11:all

   Hatch handles everything for you, including setting up an temporary
   virtual environment for each run.

-  ``pre-commit`` for all style and consistency checking. While you can
   run it with nox, this is such an important tool that it deserves to
   be installed on its own. If pre-commit fails during pushing upstream
   then stage changes, Commit Extend (into previous commit), and repeat
   pushing.

``pip``, ``pip-deepfreeze`` and ``hatch`` are pinned in
.github/workflows/constraints.txt for consistency with CI/CD.

::

   pipx install pre-commit
   pipx install pip-deepfreeze

   pacman -S python-hatch python-hyperlink python-httpx

Setting up a development with direnv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   echo "layout hatch" > .envrc
   hatch run init

Setting up a development environment manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can set up a development environment by running:

::

   python3 -m venv .venv
   source ./.venv/bin/activate
   pip install -v -e .[dev,tests,docs]

With direnv for using `Jupyter <https://jupyter.org/>`__ during
development:

::

   jupiter notebook

And only in case you need a system wide easy accessible kernel:

::

   python -m ipykernel install --user --name="clop"

Testing and coverage
~~~~~~~~~~~~~~~~~~~~

Use pytest to run the unit checks:

::

   pytest

Use ``coverage`` to generate coverage reports:

::

   coverage run --parallel -m pytest

Or use hatch:

::

   hatch run tests:all
   hatch run coverage:combine
   hatch run coverage:report

Building docs
~~~~~~~~~~~~~

You can build the docs using:

::

   hatch run docs:sync
   hatch run docs:build

You can see a preview with:

::

   hatch run docs:serve

When needed (e.g. API updates):

::

   sphinx-apidoc -f -o docs/api/ src/clophfit/

Bump and releasing
~~~~~~~~~~~~~~~~~~

To bump version and upload build to test.pypi using:

::

   hatch run bump
   hatch run bump "--increment PATCH" "--files-only" \
       ["--no-verify" to bypass pre-commit and commit-msg hooks]
   git push

while to update only the CHANGELOG.md file:

::

   hatch run ch

Release will automatically occur after pushing.

(Otherwise)

::

   pipx run --spec commitizen cz bump --changelog-to-stdout --files-only \
       (--prerelease alpha) --increment MINOR

To keep clean development history use branches and pr:

::

   gh pr create --fill
   gh pr merge --squash --delete-branch [-t “fix|ci|feat: msg”]

Configuration files
~~~~~~~~~~~~~~~~~~~

Manually updated pinned dependencies for CI/CD:

-  .github/workflows/constraints.txt (testing dependabot)

Configuration files:

-  pre-commit configured in .pre-commit-config.yaml;
-  bandit (sys) configured in bandit.yml;
-  pylint (sys) configured in pyproject.toml;
-  isort (sys) configured in pyproject.toml;
-  black configured in pyproject.toml (pinned in pre-commit);
-  ruff configured in pyproject.toml (pinned in pre-commit);
-  darglint configured in .darglint (pinned in pre-commit);
-  codespell configured in .codespellrc (pinned in pre-commit);
-  coverage configured in pyproject.toml (tests deps);
-  mypy configured in pyproject.toml (tests deps);
-  commitizen in pyproject.toml (dev deps and pinned in pre-commit).

pip-df generates requirements[-dev,docs,tests].txt.

Other manual actions:

::

   pylint src/ tests/
   bandit -r src/
