"""The whole-spectrum run reports its messages and the files it wrote."""

from __future__ import annotations

from pathlib import Path

from clophfit.prenspire import EnspireFile
from clophfit.prenspire.run import run_global

DATA = Path(__file__).parent / "EnSpire"


def test_run_global_returns_the_inspect_message_and_sidecar(tmp_path: Path) -> None:
    """``inspect`` is reported as a message and its sidecar is listed as written."""
    lines = (DATA / "NTT-G10_note.csv").read_text().splitlines()
    note = tmp_path / "NTT-G10_note.csv"
    note.write_text("\n".join([lines[0], *(ln for ln in lines[1:] if ",20.0," in ln)]))
    out = tmp_path / "out"

    result = run_global(
        EnspireFile(DATA / "G10.csv"),
        note,
        out,
        buffer=True,
        inspect=True,
    )

    sidecar = out / "NTT-G10_global_inspect.txt"
    assert sidecar.exists()
    assert sidecar in result.written
    names = {p.name for p in result.written}
    assert {"NTT-G10_K_global.csv", "NTT-G10_K_global.json"} <= names
    assert [m.text for m in result.messages] == [f"inspect: wrote {sidecar}"]
    assert [s.sample for s in result.summaries] == ["NTT-G10"]
