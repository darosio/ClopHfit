"""Nox sessions."""
import os
import sys
from pathlib import Path

import nox
from nox import Session


os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})

package = "clophfit"
locations = "src", "tests", "./noxfile.py", "docs/conf.py"
python_versions = ["3.8", "3.9", "3.10", "3.11"]
nox.options.sessions = ("lint", "mypy", "tests", "xdoctest", "typeguard", "docs")
nox.options.force_venv_backend = "venv"


@nox.session(name="lint", python=python_versions[-1])
def pre_commit(session: nox.Session) -> None:
    """Run the linter in pre-commit."""
    session.install("--constraint=.github/workflows/constraints.txt", "pre-commit")
    session.run(
        "pre-commit",
        "run",
        "--all-files",
        "--hook-stage=manual",
        "--show-diff-on-failure",
        *session.posargs,
    )


@nox.session(python=python_versions)
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or ["src", "tests", "docs/conf.py"]
    session.install(
        "--constraint=constraints.txt",
        "mypy",
        "pytest",
        "pandas-stubs",
        "types-setuptools",
        "tomli",  # avoid error in 3.11 when `nox -s mypy`
        "exceptiongroup",  # avoid error in 3.11 when `nox -s mypy`
        ".",
    )
    session.run("mypy", *args)
    if not session.posargs:
        session.run("mypy", f"--python-executable={sys.executable}", "./noxfile.py")


@nox.session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    session.install("--constraint=constraints.txt", "coverage[toml]", "pytest", ".")
    # session.run("pdm", "install", "-G", "tests", external=True)
    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@nox.session(python=python_versions[-1])
def coverage(session: Session) -> None:
    """Produce the coverage report."""
    args = session.posargs or ["report"]
    session.install("--constraint=constraints.txt", "coverage[toml]")
    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")
    session.run("coverage", *args)


@nox.session(python=python_versions)
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    session.install("--constraint=constraints.txt", "xdoctest", "pygments", ".")
    session.run("python", "-m", "xdoctest", package, *args)


@nox.session(python=python_versions[-1])
def typeguard(session: Session) -> None:
    """Runtime type checking using Typeguard."""
    session.install(
        "--constraint=constraints.txt", "pytest", "typeguard", "pygments", "."
    )
    session.run("pytest", f"--typeguard-packages={package}", *session.posargs)


@nox.session(python=python_versions[-1])
def docs(session: Session) -> None:
    """Build the docs. Pass "serve" to serve."""
    session.install(
        "--constraint=constraints.txt",
        "sphinx",
        "sphinx-click",
        "pydata_sphinx_theme",
        "myst-parser",
        "nbsphinx",
        "sphinxcontrib-plantuml",
        "sphinx-autodoc-typehints",
        "autodocsumm",
        ".",
    )
    session.run("sphinx-build", "docs", "docs/_build")
    if session.posargs:
        if "serve" in session.posargs:
            print("Launching docs at http://localhost:8000/ - use Ctrl-C to quit")
            session.run("python", "-m", "http.server", "8000", "-d", "docs/_build")
        else:
            session.warn("Unsupported argument to docs")


@nox.session
def clean(session: Session) -> None:
    """Clean local repository."""
    session.run(
        "rm",
        "-rf",
        ".coverage",
        "./__pycache__",
        "./.nox",
        "./.mypy_cache",
        "./.pytest_cache",
        "./docs/_build",
        "./src/" + package + "/__pycache__",
        "./tests/__pycache__",
        "./dist",
        external=True,
    )


@nox.session
def bump(session: Session) -> None:
    """Bump repository and upload to testpypi."""
    args = session.posargs  # or ["--increment", "PATCH"]
    session.install("--constraint=constraints.txt", "commitizen")
    session.run(
        "cz",
        "bump",
        "--major-version-zero",
        "-ch",
        # "--files-only",
        # "--no-verify",  # bypass pre-commit and commit-msg hooks
        *args,
    )
    session.run("pdm", "publish", "-r", "testpypi", external=True)


@nox.session
def ch(session: Session) -> None:
    """Bump repository and upload to testpypi."""
    args = session.posargs or ["HEAD"]
    session.install("--constraint=constraints.txt", "commitizen")
    session.run("cz", "ch", "--incremental", "--unreleased-version", *args)


@nox.session
def init(session: nox.Session) -> None:
    """Install pre-commit hooks."""
    args = session.posargs
    session.run("pre-commit", "install", *args, external=True)
    session.run(
        "pre-commit",
        "install",
        "--hook-type",
        "commit-msg",
        "--hook-type",
        "pre-push",
        external=True,
    )
