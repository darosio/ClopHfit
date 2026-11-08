"""Golden characterisation of ``ppr enspire`` and ``ppr enspire-batch``.

Every output is compared with what the commands wrote before any refactoring:

- ``.csv`` and ``.json`` files are compared byte-for-byte with the copies in
  ``tests/golden/enspire/<case>/``.
- ``.pdf`` and ``.png`` files are compared by SHA-256 of their bytes, with the
  PDF creation and modification dates removed, against ``manifest.json``.
- ``.nc`` traces are not compared: their bytes differ between identical seeded
  runs, and the posterior results they hold are covered by the CSV files.

The Bayesian modes run with ``--draws 200 --chains 2`` to keep the test short;
``global-bayes-marginal`` is left out until its PyMC model is settled.

Regenerate after an intended change with::

    CLOPHFIT_REGEN_GOLDEN=1 uv run pytest -m slow tests/test_enspire_golden.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from clophfit.__main__ import ppr

pytestmark = pytest.mark.slow

TESTS = Path(__file__).parent
DATA = TESTS / "EnSpire"
GOLDEN = TESTS / "golden" / "enspire"
REGEN = os.environ.get("CLOPHFIT_REGEN_GOLDEN") == "1"
PDF_DATES = re.compile(rb"/(CreationDate|ModDate) \([^)]*\)")
BAYES_FAST = ["--draws", "200", "--chains", "2"]
X_SIGMA = ["--x-sigma", "0.1"]
TEXT_SUFFIXES = {".csv", ".json"}
HASHED_SUFFIXES = {".pdf", ".png"}
MANIFEST = "manifest.json"
# provenance keys that depend on the installed library versions, not on the fit
PROVENANCE_ENVIRONMENT = ("clophfit_version", "environment")
# SVD figures are not reproducible run to run (their pixels change between
# identical runs); their CSV and JSON outputs are still compared exactly.
UNSTABLE_FIGURES = {"svd", "both"}


def _data_csv(folder: Path) -> Path:
    """Copy the G10 plate into ``folder`` (the note is named after it)."""
    target = folder / "G10.csv"
    shutil.copyfile(DATA / "G10.csv", target)
    return target


def _note(folder: Path, *, subset: bool) -> Path:
    """Copy the G10 note; ``subset`` keeps the 20 °C wells only."""
    source = (DATA / "NTT-G10_note.csv").read_text().splitlines()
    note = folder / "NTT-G10_note.csv"
    if subset:
        kept = [line for line in source[1:] if ",20.0," in line]
        note.write_text("\n".join([source[0], *kept]) + "\n")
    else:
        note.write_text("\n".join(source) + "\n")
    return note


def _enspire_args(data: Path, note: Path, method: list[str]) -> list[str]:
    return ["enspire", str(data), str(note), *method]


CASES: dict[str, tuple[list[str], bool, list[str]]] = {
    # name: (method options, 20 C wells only, extra options)
    "band": (["--method", "band"], True, []),
    "svd": (["--method", "svd"], True, []),
    "both": (["--method", "both"], True, []),
    "global": (["--method", "global"], True, []),
    "bayes": (["--method", "bayes"], True, BAYES_FAST),
    "global-bayes": (["--method", "global-bayes"], True, [*X_SIGMA, *BAYES_FAST]),
    "vanthoff": (["--method", "vanthoff"], False, []),
    "vanthoff-bayes": (
        ["--method", "vanthoff-bayes"],
        False,
        [*X_SIGMA, *BAYES_FAST],
    ),
}


def _run(out: Path, args: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    result = CliRunner().invoke(ppr, ["--out", str(out), *args])
    assert result.exit_code == 0, result.output


def _outputs(out: Path) -> list[Path]:
    return sorted(
        p
        for p in out.iterdir()
        if p.is_file() and p.suffix != ".nc" and p.name != "log.txt"
    )


def _normalised_bytes(path: Path) -> bytes:
    """Bytes to compare: JSON minus the environment-dependent provenance keys."""
    if path.suffix == ".json":
        record = json.loads(path.read_text(encoding="utf-8"))
        for key in PROVENANCE_ENVIRONMENT:
            record.pop(key, None)
        return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    return path.read_bytes()


def _hash_normalised(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix == ".pdf":
        data = PDF_DATES.sub(b"", data)
    return hashlib.sha256(data).hexdigest()


def _check(case: str, out: Path) -> None:
    folder = GOLDEN / case
    manifest_path = folder / MANIFEST
    outputs = _outputs(out)
    hashed = (
        {}
        if case in UNSTABLE_FIGURES
        else {
            p.name: _hash_normalised(p) for p in outputs if p.suffix in HASHED_SUFFIXES
        }
    )
    texts = [p for p in outputs if p.suffix in TEXT_SUFFIXES]
    if REGEN:
        folder.mkdir(parents=True, exist_ok=True)
        for p in texts:
            (folder / p.name).write_bytes(_normalised_bytes(p))
        manifest_path.write_text(json.dumps(hashed, indent=2, sort_keys=True) + "\n")
        return
    golden_names = sorted(
        p.name
        for p in folder.iterdir()
        if p.suffix in TEXT_SUFFIXES and p.name != MANIFEST
    )
    assert golden_names == sorted(p.name for p in texts), (
        f"{case}: output files changed; regenerate with CLOPHFIT_REGEN_GOLDEN=1"
    )
    for p in texts:
        golden = folder / p.name
        assert _normalised_bytes(p) == golden.read_bytes(), f"{case}: {p.name} differs"
    expected = json.loads(manifest_path.read_text())
    assert hashed == expected, f"{case}: pdf/png outputs differ"


@pytest.fixture(autouse=True)
def _seeded_confidence_bands(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed the unseeded Monte Carlo bands drawn by ``fitting.plotting``.

    The bands are plotted, not written to any table, so an unseeded draw makes
    the PDFs differ between identical runs. This keeps the golden figures stable
    without touching the library code.
    """
    real = np.random.default_rng
    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda seed=None: real(0 if seed is None else seed),
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_enspire_matches_golden(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each EnSpire method writes the same CSV, JSON and figure outputs as before."""
    method, subset, extra = CASES[case]
    inputs = tmp_path / "in"
    inputs.mkdir()
    _data_csv(inputs)
    note = _note(inputs, subset=subset)
    monkeypatch.chdir(inputs)
    out = tmp_path / "out"
    _run(
        out, [*_enspire_args(Path("G10.csv"), note.relative_to(inputs), method), *extra]
    )
    _check(case, out)


def test_enspire_batch_matches_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``enspire-batch`` writes the same summary tables as before."""
    root = tmp_path / "batch" / "G10"
    root.mkdir(parents=True)
    _data_csv(root)
    _note(root, subset=True)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    _run(out, ["enspire-batch", "batch"])
    _check("batch", out)
