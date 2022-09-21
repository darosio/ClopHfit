"""Test ``clop`` cli."""
from __future__ import annotations

from click.testing import CliRunner

from clophfit.__main__ import clop


def test_eq1() -> None:
    """It runs XXX pr.tecan and generates correct results and graphs."""
    runner = CliRunner()
    result = runner.invoke(clop, ["eq1", "2", "2", "2"])
    assert result.exit_code == 0
    assert "4." in result.output
