"""The command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from detbench.cli import build_parser, main


def test_demo_runs_without_assets(capsys):
    assert main(["--demo", "--demo-images", "6"]) == 0
    out = capsys.readouterr().out
    assert "Average Precision" in out
    assert "error taxonomy" in out
    assert "per-stage latency" in out


def test_demo_subcommand_matches_the_flag(capsys):
    assert main(["demo", "--demo-images", "6"]) == 0
    assert "synthetic" in capsys.readouterr().out


def test_demo_states_that_the_numbers_are_synthetic(capsys):
    main(["--demo", "--demo-images", "4"])
    out = capsys.readouterr().out.lower()
    assert "synthetic" in out
    assert "not coco" in out


def test_demo_writes_figures(tmp_path: Path, capsys):
    pytest.importorskip("matplotlib")
    assert main(["demo", "--demo-images", "6", "--figures", str(tmp_path)]) == 0
    produced = sorted(p.name for p in tmp_path.glob("*.png"))
    assert produced == [
        "demo-errors.png",
        "demo-per-class-ap.png",
        "demo-pr-curves.png",
        "demo-threshold-sweep.png",
    ]
    assert all(p.stat().st_size > 1000 for p in tmp_path.glob("*.png"))


def test_no_arguments_prints_help_and_fails(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out


def test_parser_exposes_every_documented_subcommand():
    parser = build_parser()
    actions = [
        a for a in parser._actions if getattr(a, "choices", None)
        and "evaluate" in getattr(a, "choices", {})
    ]
    assert actions, "no subparser action found"
    assert set(actions[0].choices) == {
        "demo", "evaluate", "quantize", "analyse", "profile"
    }


def test_evaluate_requires_a_model():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--annotations", "a", "--images", "b"])


def test_version_flag_exits_cleanly():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0
