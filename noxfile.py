"""Nox sessions."""
import os
import shlex
from pathlib import Path
from textwrap import dedent

import nox
import nox_poetry
from nox_poetry.sessions import Session


package = "clophfit"
locations = "src", "tests", "./noxfile.py", "docs/conf.py"
python_versions = ["3.8", "3.9", "3.10"]
nox.options.sessions = "pre-commit", "mypy", "tests", "xdoctest", "docs"
# nox.options.sessions = "pre-commit", "safety", "mypy", "tests", "typeguard", "docs"


def activate_virtualenv_in_precommit_hooks(session: Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.

    Parameters
    ----------
    session : Session
        The Session object.

    """
    assert session.bin is not None  # noqa: S101

    # Only patch hooks containing a reference to this session's bindir. Support
    # quoting rules for Python and bash, but strip the outermost quotes so we
    # can detect paths within the bindir, like <bindir>/python.
    bindirs = [
        bindir[1:-1] if bindir[0] in "'\"" else bindir
        for bindir in (repr(session.bin), shlex.quote(session.bin))
    ]

    virtualenv = session.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    headers = {
        # pre-commit < 2.16.0
        "python": f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {session.bin!r},
                os.environ.get("PATH", ""),
            ))
            """,
        # pre-commit >= 2.16.0
        "bash": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(session.bin)}"{os.pathsep}$PATH"
            """,
        # pre-commit >= 2.17.0 on Windows forces sh shebang
        "/bin/sh": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(session.bin)}"{os.pathsep}$PATH"
            """,
    }

    hookdir = Path(".git") / "hooks"
    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        if not hook.read_bytes().startswith(b"#!"):
            continue

        text = hook.read_text()

        if not any(
            Path("A") == Path("a") and bindir.lower() in text.lower() or bindir in text
            for bindir in bindirs
        ):
            continue

        lines = text.splitlines()

        for executable, header in headers.items():
            if executable in lines[0].lower():
                lines.insert(1, dedent(header))
                hook.write_text("\n".join(lines))
                break


@nox_poetry.session(name="pre-commit", python=python_versions[-1])
def precommit(session: Session) -> None:
    """Lint using pre-commit."""
    args = session.posargs or [
        "run",
        "--all-files",
        "--hook-stage=manual",
        "--show-diff-on-failure",
    ]
    session.install(
        "black",
        "darglint",
        "flake8",
        "flake8-bandit",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-eradicate",
        "flake8-rst-docstrings",
        "flake8-pytest-style",
        "flake8-comprehensions",
        "isort",
        "pep8-naming",
        "pre-commit",
        "pre-commit-hooks",
        "pyupgrade",
        "commitizen",
    )
    # TODO: other linters session.run("rst-lint", "README.rst")  # for PyPI readme.rst
    session.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)


@nox_poetry.session(python=["3.10"])
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    session.install("mypy")
    session.run("mypy", *args)


@nox_poetry.session(python=["3.10", "3.9", "3.8"])
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs or ["--cov", "-v"]
    session.install("coverage[toml]", "pytest", "pytest-cov", ".")
    session.run("pytest", *args)


@nox_poetry.session(python=["3.10", "3.9", "3.8"])
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    session.install("xdoctest", "pygments", ".")
    session.run("echo", "== Xdoctest is testing: ", package, "and", *args, "==")
    session.run("python", "-m", "xdoctest", package, *args)


@nox_poetry.session(python=python_versions[-1])
def docs(session: Session) -> None:
    """Build the documentation."""
    session.install(
        "sphinx",
        "sphinx-click",
        "pydata_sphinx_theme",
        "myst-parser",
        "sphinxcontrib-plantuml",
        "sphinx-autodoc-typehints",
        "nbsphinx",
        ".",
    )
    session.run("sphinx-build", "docs", "docs/_build")


@nox_poetry.session(python="3.10")
def clean(session: Session) -> None:
    """Clean local repository."""
    session.run(
        "rm",
        "-r",
        "./README.tmp.html",
        "./__pycache__",
        "./.nox",
        "./.mypy_cache",
        "./.pytest_cache",
        "./docs/_build",
        "./src/clophfit/__pycache__",
        "./tests/__pycache__",
        "./dist",
        external=True,
    )
