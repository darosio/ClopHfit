"""Nox sessions."""
import os
import shlex
import sys
from pathlib import Path
from textwrap import dedent

import nox
import nox_poetry
from nox_poetry.sessions import Session


package = "clophfit"
locations = "src", "tests", "./noxfile.py", "docs/conf.py"
python_versions = ["3.8", "3.9", "3.10"]
nox.options.sessions = (
    "pre-commit",
    "safety",
    "mypy",
    "tests",
    "xdoctest",
    "typeguard",
    "docs",
)


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
    session.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)


@nox_poetry.session(python=python_versions[-1])
def safety(session: Session) -> None:
    """Scan dependencies for insecure packages."""
    requirements = session.poetry.export_requirements()
    session.install("safety")
    session.run("safety", "check", "--full-report", f"--file={requirements}")


@nox_poetry.session(python=python_versions)
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or ["src", "tests", "docs/conf.py"]
    session.run(
        "rm", "-rf", ".mypy_cache/", external=True
    )  # for types-jinja2 from pyparser
    session.install("mypy", "pytest", "pandas-stubs", "types-setuptools", ".")
    session.run("mypy", *args)
    if not session.posargs:
        session.run("mypy", f"--python-executable={sys.executable}", "./noxfile.py")


@nox_poetry.session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    session.install("coverage[toml]", "pytest", ".")  # "pytest-cov"
    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@nox_poetry.session(python=python_versions[-1])
def coverage(session: Session) -> None:
    """Produce the coverage report."""
    args = session.posargs or ["report"]

    session.install("coverage[toml]")

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", *args)


@nox_poetry.session(python=python_versions)
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    session.install("xdoctest", "pygments", ".")
    session.run("echo", "== Xdoctest is testing: ", package, "and", *args, "==")
    session.run("python", "-m", "xdoctest", package, *args)


@nox_poetry.session(python=python_versions[-1])
def typeguard(session: Session) -> None:
    """Runtime type checking using Typeguard."""
    session.install(".")
    session.install("pytest", "typeguard", "pygments")
    session.run("pytest", f"--typeguard-packages={package}", *session.posargs)


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
        "autodocsumm",
        ".",
    )
    session.run("sphinx-build", "docs", "docs/_build")


@nox_poetry.session(python=python_versions[-1])
def clean(session: Session) -> None:
    """Clean local repository."""
    session.run(
        "rm",
        "-rf",
        ".coverage" "./__pycache__",
        "./.nox",
        "./.mypy_cache",
        "./.pytest_cache",
        "./docs/_build",
        "./src/" + package + "/__pycache__",
        "./tests/__pycache__",
        "./dist",
        external=True,
    )


@nox_poetry.session(python=python_versions[-1])
def bump(session: Session) -> None:
    """Bump repository and upload to testpypi."""
    session.install("commitizen")
    session.run(
        "cz",
        "bump",
        "-ch",
        "--files-only",
        # "--no-verify",
        "--increment",
        "PATCH",
        # "-pr",
        # "rc",
    )
    session.run("poetry", "publish", "-r", "testpypi", "--build", external=True)


# https://nox.thea.codes/en/stable/cookbook.html?highlight=input#the-auto-release
