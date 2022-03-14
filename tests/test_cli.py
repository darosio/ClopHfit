"""Test ``clop`` cli."""

from typer.testing import CliRunner

from clophfit.cli import app


def test_eq1_cli() -> None:
    """It runs XXX pr.tecan and generates correct results and graphs."""
    runner = CliRunner()
    result = runner.invoke(app, ["eq1", "2", "2", "2"])
    assert result.exit_code == 0
    assert "4." in result.output
