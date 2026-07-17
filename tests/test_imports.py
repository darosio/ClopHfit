"""Cold-import regression tests.

Each import runs in a fresh interpreter so partially-initialized-package
regressions surface instead of being masked by an already-populated
``sys.modules``.
"""

from __future__ import annotations

import subprocess  # noqa: S404
import sys


def _cold_import(statement: str) -> None:
    subprocess.run([sys.executable, "-c", statement], check=True)  # noqa: S603


def test_cold_import_fitting_package() -> None:
    """Importing the package triggers the eager ``__init__`` chain."""
    _cold_import("import clophfit.fitting")


def test_cold_import_data_structures_first() -> None:
    """Importing the submodule first must not hit a partial parent package."""
    _cold_import("import clophfit.fitting.data_structures")


def test_cold_import_prtecan() -> None:
    """Prtecan imports fitting at module level; it must stay importable."""
    _cold_import("import clophfit.prtecan")
