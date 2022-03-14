"""Nox sessions."""
import nox
import nox_poetry
from nox_poetry.sessions import Session

package = "clophfit"
nox.options.sessions = "lint", "mypy", "tests", "xdoctest", "docs"
locations = ("src", "tests", "noxfile.py", "docs/conf.py")


@nox_poetry.session(python="3.10")
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    session.install("black[jupyter]")
    session.run("echo", "== Black is formatting: ", *args, " ==", external=True)
    session.run("black", "--exclude", "src/clophfit/old", "docs", *args)


@nox_poetry.session(python="3.10")
def lint(session: Session) -> None:
    """Lint using flake8."""
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-isort",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-rst-docstrings",
        "flake8-rst",
        "darglint",
        "flake8-eradicate",
        "flake8-comprehensions",
        "flake8-pytest-style",
        "flake8-annotations",
    )
    session.run("flake8", "--exclude", "src/clophfit/old", *args)
    # session.run("flake8", "--exclude", "src/clophfit/old", "--select", "RST", *args)
    # session.run("flake8-rst", "--exclude", "src/clophfit/old", *args)
    session.run("rst-lint", "README.rst")  # for PyPI readme.rst
    # session.run("rst-lint", "CONTRIBUTING.rst")  # for PyPI readme.rst


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


@nox_poetry.session(python=["3.10"])
def docs(session: Session) -> None:
    """Build the documentation."""
    session.install(
        "sphinx",
        "sphinxcontrib-plantuml",
        "pydata-sphinx-theme",
        "sphinx-autodoc-typehints",
        "nbsphinx",
        ".",
    )
    session.run("sphinx-build", "docs", "docs/_build")
