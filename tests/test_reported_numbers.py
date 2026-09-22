"""Check that every number quoted in README.md still matches the file it came from.

The result tables and figures in ``output/`` are committed, so this can run
without any cohort data. It exists because the README restates results that live
in five different files: an edit that updates one and not the other would
otherwise reach a reader as a contradiction.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()


def sheet(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_excel(ROOT / "output" / name, **kwargs)


def readme_table_row(label: str) -> list[str]:
    """Return the cells of the markdown table row beginning with ``label``."""
    for line in README.splitlines():
        if line.startswith(f"| {label}"):
            return [cell.strip() for cell in line.strip("|").split("|")]
    raise AssertionError(f"no README table row starting with {label!r}")


# --- cohort counts ------------------------------------------------------------

@pytest.mark.parametrize(
    "label, workbook, sheet_name",
    [
        ("NHANES 1999–2012", "stats.xlsx", "NHANES"),
        ("KNHANES 2019–2021", "stats.xlsx", "KNHANES"),
        ("Pooled training set", "stats.xlsx", "COMBINE"),
        ("Taiwan Biobank (external)", "stats_twb.xlsx", "TWB(predict)"),
    ],
)
def test_cohort_counts_match_stats_workbook(label, workbook, sheet_name):
    analysed, positives = readme_table_row(label)[2:4]
    counts = sheet(workbook, sheet_name=sheet_name)
    row = counts[counts[counts.columns[0]] == "N"].iloc[0]

    assert analysed == row["All"], f"{label}: analysed n"

    count, percent = re.match(r"\*?\*?([\d,]+)\*?\*? \((\d+\.\d)%\)", positives).groups()
    assert count == row["IR+"], f"{label}: IR+ count"

    expected = 100 * int(row["IR+"].replace(",", "")) / int(row["All"].replace(",", ""))
    assert float(percent) == pytest.approx(expected, abs=0.05), f"{label}: IR+ percentage"


# --- pooled model performance -------------------------------------------------

@pytest.mark.parametrize(
    "label, feature_set, model",
    [
        ("CatBoost", "all features (241)", "CatBoost"),
        ("Soft voting (3 models)", "all features (241)", "Voting"),
        ("**CatBoost, top 20 by SHAP**", "top 20 by SHAP", "CatBoost"),
    ],
)
def test_model_metrics_match_performance_workbook(label, feature_set, model):
    cells = readme_table_row(label)
    performance = sheet("final_model_performance.xlsx")
    row = performance[
        (performance.feature_set == feature_set) & (performance.model == model)
    ].iloc[0]

    columns = ["roc_auc", "accuracy", "sensitivity(recall)", "specificity", "NPV", "pr_auc"]
    for quoted, column in zip(cells[2:8], columns):
        assert float(quoted.strip("*")) == pytest.approx(row[column]), f"{label}: {column}"


def test_deployed_model_matches_its_manifest():
    """The README's headline model is the one actually saved in models/."""
    manifest = pd.read_csv(ROOT / "models" / "manifest.csv")
    top20 = manifest[manifest.file == "catboost_top20.pkl"].iloc[0]
    cells = readme_table_row("**CatBoost, top 20 by SHAP**")

    assert int(cells[1].strip("*")) == top20.n_features
    assert float(cells[2].strip("*")) == pytest.approx(top20.roc_auc)
    assert float(cells[4].strip("*")) == pytest.approx(top20["sensitivity(recall)"])
    assert float(cells[6].strip("*")) == pytest.approx(top20.NPV)


# --- cross-cohort transfer ----------------------------------------------------

def test_cross_cohort_auc_ranges_match_workbook():
    """Both AUC ranges and both within-cohort ceilings, as quoted in the prose."""
    transfer = sheet("cross_ethnic.xlsx")
    cross = transfer[transfer.evaluation == "cross-cohort"]
    within = transfer[transfer.evaluation == "within-cohort hold-out"]

    quoted = re.search(
        r"AUC is (\d\.\d+)–(\d\.\d+) sending NHANES to KNHANES and\s+"
        r"(\d\.\d+)–(\d\.\d+) sending KNHANES to NHANES, against within-cohort ceilings of "
        r"(\d\.\d+) and (\d\.\d+)",
        README,
    )
    assert quoted, "the cross-cohort sentence no longer has the expected shape"
    to_knhanes_low, to_knhanes_high, to_nhanes_low, to_nhanes_high, nhanes_ceiling, knhanes_ceiling = (
        float(value) for value in quoted.groups()
    )

    outgoing = cross[cross.trained_on == "NHANES"].roc_auc
    incoming = cross[cross.trained_on == "KNHANES"].roc_auc
    assert (outgoing.min(), outgoing.max()) == (to_knhanes_low, to_knhanes_high)
    assert (incoming.min(), incoming.max()) == (to_nhanes_low, to_nhanes_high)

    assert within[within.trained_on == "NHANES"].roc_auc.max() == nhanes_ceiling
    assert within[within.trained_on == "KNHANES"].roc_auc.max() == knhanes_ceiling


def test_every_algorithm_is_scored_in_both_directions():
    """Direction and algorithm must not be confounded, which is what the prose claims."""
    cross = sheet("cross_ethnic.xlsx").query("evaluation == 'cross-cohort'")
    by_direction = cross.groupby("trained_on").model.apply(set)

    assert set(by_direction.index) == {"NHANES", "KNHANES"}
    assert by_direction["NHANES"] == by_direction["KNHANES"]
    assert len(by_direction["NHANES"]) == 4


# --- differential methylation -------------------------------------------------

def test_methylation_counts_match_the_probe_table():
    probes = sheet("met-point.xlsx")
    assert f"{len(probes)} differ between the predicted groups" in README

    symbols = {
        symbol
        for name in probes["UCSC_REFGENE_NAME"].dropna()
        for symbol in str(name).split(";")
        if symbol
    }
    # UBE2QP1 is an alias of UBE2Q2P1, which the README states explicitly.
    genes = {"UBE2Q2P1" if symbol == "UBE2QP1" else symbol for symbol in symbols}
    assert f"spanning {len(genes)} named" in README
    assert f"with {len(symbols)} symbols" in README


def test_probe_cascade_is_internally_consistent():
    """The filter cascade in the README must end at the number of tests reported."""
    cascade = re.search(r"That cascade is ([\d,]+) → ([\d,]+) → ([\d,]+) → \*\*([\d,]+)\*\*", README)
    assert cascade, "the probe cascade sentence no longer has the expected shape"
    counts = [int(value.replace(",", "")) for value in cascade.groups()]

    assert counts == sorted(counts, reverse=True), "a filter cannot increase the probe count"
    assert counts[0] == 866_895, "the EPIC manifest size is fixed"

    tested = sheet("met-point.xlsx")
    assert (tested["CHR"].astype(str).isin(["X", "Y"])).sum() == 0, "sex chromosomes are excluded"


# --- figures ------------------------------------------------------------------

def test_every_figure_is_accounted_for():
    """No orphan PNG in output/, and no README reference to a figure that is gone."""
    produced = {path.stem for path in (ROOT / "output").glob("*.png")}

    # A figure may be named directly, or be a sibling of one that is: the
    # colour-blind variants and the panel-labelled copies carry these suffixes.
    def base(stem: str) -> str:
        for suffix in ("_color_blind", "_a", "_b"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    undocumented = {stem for stem in produced if base(stem) not in README}
    assert not undocumented, f"figures in output/ that the README never mentions: {sorted(undocumented)}"

    # A leading underscore is the README naming a suffix, not a file.
    for referenced in re.findall(r"`([A-Za-z0-9+][A-Za-z0-9_+\-]*)\.png`", README):
        assert (ROOT / "output" / f"{referenced}.png").exists(), f"README names a missing figure: {referenced}"


# --- uncertainty and calibration -----------------------------------------------

def test_quoted_intervals_match_the_bootstrap_workbook():
    """Each interval the README quotes, and the claim that they all overlap."""
    intervals = sheet("final_model_ci.xlsx")
    deployed = intervals[
        (intervals.feature_set == "top 20 by SHAP") & (intervals.model == "CatBoost")
    ].set_index("metric")

    quoted = re.findall(r"(AUC|sensitivity|NPV) (\d\.\d+) \((\d\.\d+)–(\d\.\d+)\)", README)
    assert len(quoted) == 3, "the uncertainty sentence no longer quotes three intervals"

    for name, estimate, lower, upper in quoted:
        metric = {"AUC": "roc_auc", "sensitivity": "sensitivity(recall)", "NPV": "NPV"}[name]
        row = deployed.loc[metric]
        assert (float(estimate), float(lower), float(upper)) == (
            row.estimate,
            row.ci_lower,
            row.ci_upper,
        ), f"{name} interval"

    auc = intervals[intervals.metric == "roc_auc"]
    assert auc.ci_lower.max() <= auc.ci_upper.min(), "the README says every AUC interval overlaps"


def test_every_interval_contains_its_estimate():
    intervals = sheet("final_model_ci.xlsx")
    outside = intervals[
        (intervals.estimate < intervals.ci_lower) | (intervals.estimate > intervals.ci_upper)
    ]
    assert outside.empty, f"intervals not containing their estimate:\n{outside}"


def test_quoted_calibration_matches_the_workbook():
    summary = sheet("calibration.xlsx", sheet_name="summary")
    deployed = summary[summary.model == "CatBoost (top 20)"].iloc[0]
    full = summary[summary.model == "CatBoost (241 features)"].iloc[0]

    slope = re.search(r"calibration slope is\s+(\d\.\d+)", README)
    intercept = re.search(r"the intercept is −(\d\.\d+)", README)
    brier = re.search(r"Brier score (\d\.\d+)", README)
    assert slope and intercept and brier, "the calibration sentence no longer has the expected shape"

    assert float(slope.group(1)) == full.calibration_slope
    assert -float(intercept.group(1)) == full.calibration_intercept
    assert float(brier.group(1)) == full.brier_score == deployed.brier_score


def test_calibration_bins_cover_the_test_set():
    """The deciles must account for every participant, at equal size."""
    bins = sheet("calibration.xlsx", sheet_name="CatBoost (top 20)")
    performance = sheet("final_model_performance.xlsx")
    row = performance[performance.feature_set == "top 20 by SHAP"].iloc[0]

    assert bins.n.sum() == row.TP + row.TN + row.FP + row.FN
    assert bins.n.nunique() == 1, "quantile bins should hold equal numbers"
