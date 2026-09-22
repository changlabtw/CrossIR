"""Unit tests that need no cohort data.

No individual-level data is distributed with this repository, so these exercise
the transformations on small constructed frames instead. They cover the places
where a silent change would move a published number: the label definition, the
feature generator's column count, the target-leak drop, and the multiple-testing
correction.
"""

import numpy as np
import polars as pl
import pytest

from src.data.io import HOMAIR_CUTOFF, add_homair
from src.features.interactions import generate_interactions
from src.features.selection import drop_target_derived
from src.methylation.differential import adjust_pvalues
from src.viz.figures import add_panel_label


def test_homair_formula_and_label_boundary():
    """HOMA-IR is insulin x glucose / 405, and the label is strictly above 2.5."""
    # Chosen so the three rows land just below, exactly on, and just above 2.5.
    frame = pl.DataFrame(
        {
            "FASTING_INSULIN": [10.0, 10.0, 10.0],
            "FASTING_GLUCOSE": [101.0, 101.25, 102.0],
        }
    )
    result = add_homair(frame)

    assert result["HOMA-IR"].to_list() == pytest.approx([10 * g / 405 for g in (101.0, 101.25, 102.0)])
    assert result["HOMA-IR"][1] == HOMAIR_CUTOFF
    assert result["IR"].to_list() == [False, False, True], "the cut-off must be exclusive"


def test_generate_interactions_counts_and_race_constant():
    """Every pair and every unary transform is generated, and RACE is a constant."""
    frame = pl.DataFrame({"AGE": [40.0, 50.0], "BMI": [22.0, 27.0], "TG": [90.0, 150.0]})
    result = generate_interactions(frame, race=2)

    # AGE is excluded from interactions, so BMI and TG are the two combinable
    # columns: 1 product + 1 ratio, plus log, sqrt and square of each.
    assert result["RACE"].to_list() == [2, 2]
    assert result.width == 3 + 1 + 2 + 3 * 2
    assert result.columns == sorted(result.columns), "column order is load-bearing"


def test_drop_target_derived_removes_every_leak():
    frame = pl.DataFrame(
        {
            "Release_No": [1],
            "AGE": [40.0],
            "HOMA-IR": [3.0],
            "log(HOMA-IR)": [1.1],
            "FASTING_INSULIN_mul_AGE": [4.0],
            "DIABETES": [0],
            "IR": [True],
        }
    )
    assert drop_target_derived(frame).columns == ["AGE", "IR"]


def test_benjamini_hochberg_against_hand_computed_values():
    """BH on a known vector, checked against the closed form."""
    raw = [0.001, 0.008, 0.039, 0.041, 0.042]
    adjusted = adjust_pvalues(pl.DataFrame({"p": raw}), columns=["p"])["Adj. p"].to_list()

    n = len(raw)
    expected = np.minimum.accumulate(
        [p * n / rank for p, rank in zip(reversed(raw), range(n, 0, -1))]
    )[::-1]
    assert adjusted == pytest.approx(list(expected))
    assert all(a >= p for a, p in zip(adjusted, raw)), "correction never lowers a p-value"


def test_add_panel_label_covers_nothing(tmp_path):
    """The label goes in an added strip; the original pixels survive untouched."""
    from PIL import Image

    source = tmp_path / "plot.png"
    original = Image.new("RGB", (120, 80), (255, 255, 255))
    original.putpixel((0, 79), (12, 34, 56))  # a corner of the "plot" to protect
    original.save(source)

    labelled = Image.open(add_panel_label(source, "(a)", tmp_path / "plot_a.png"))
    strip = labelled.height - original.height

    assert strip > 0
    assert labelled.width == original.width
    assert list(labelled.crop((0, strip, 120, strip + 80)).getdata()) == list(original.getdata())
