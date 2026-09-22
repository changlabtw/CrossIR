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
    "label, sheet_name",
    [
        ("NHANES 1999–2012", "NHANES"),
        ("KNHANES 2019–2021", "KNHANES"),
        ("Pooled training set", "COMBINE"),
        ("Taiwan Biobank (external)", "TWB(predict)"),
    ],
)
def test_cohort_counts_match_stats_workbook(label, sheet_name):
    analysed, positives = readme_table_row(label)[2:4]
    counts = sheet("stats.xlsx", sheet_name=sheet_name)
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
