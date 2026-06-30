"""Tests for the structural-comment check (check_banners, code-style §1-§2)."""

from __future__ import annotations

import check_banners

from pathlib import Path

PATH = Path("sample.py")


def test_consistent_banner_passes() -> None:
    lines = [
        "#" * 30,
        "#" + "Section".center(28) + "#",
        "#" * 30,
    ]
    assert check_banners.check_banners(PATH, lines) is False


def test_mixed_widths_flagged() -> None:
    lines = [
        "#" * 30,
        "#" + "A".center(28) + "#",
        "#" * 30,
        "",
        "#" * 20,
        "#" + "B".center(18) + "#",
        "#" * 20,
    ]
    assert check_banners.check_banners(PATH, lines) is True


def test_label_not_centred_flagged() -> None:
    # Border 30 wide but the label is jammed left -> not centred.
    lines = [
        "#" * 30,
        "#" + "Label".ljust(28) + "#",
        "#" * 30,
    ]
    assert check_banners.check_banners(PATH, lines) is True


def test_banner_indent_must_match_headed_code() -> None:
    # Banner at column 0, but the code it heads is indented 4 -> flagged.
    lines = [
        "#" * 30,
        "#" + "Section".center(28) + "#",
        "#" * 30,
        "",
        "    x = 1",
    ]
    assert check_banners.check_banners(PATH, lines) is True


def test_intro_detached_below_flagged() -> None:
    lines = ["# ── Step ──────", "", "x = 1"]
    assert check_banners.check_intros(PATH, lines) is True


def test_intro_well_formed_passes() -> None:
    lines = ["def f():", "    # ── Step ──────", "    x = 1"]
    assert check_banners.check_intros(PATH, lines) is False
