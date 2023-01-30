"""Test ``clop`` cli."""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from clophfit.__main__ import clop


def test_eq1() -> None:
    """It runs XXX pr.tecan and generates correct results and graphs."""
    runner = CliRunner()
    result = runner.invoke(clop, ["eq1", "2", "2", "2"])
    assert result.exit_code == 0
    assert "4." in result.output


@pytest.mark.filterwarnings("ignore:OVER")
def test_prtecan(tmp_path: Path) -> None:
    """It runs XXX pr.tecan and generates correct results and graphs."""
    out = tmp_path / "out3"
    out.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        clop,
        [
            "prtecan",
            "tests/Tecan/140220/list.pH",
            "--out",
            str(out),
            "--fit",
            "--scheme",
            "tests/Tecan/140220/scheme.txt",
            "--bg",
        ],
    )
    assert result.exit_code == 0
