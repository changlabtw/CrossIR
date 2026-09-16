"""Feature-set construction for the modelling stages.

The four nested feature sets defined here are what the ablation compares, and
the largest of them is the feature matrix the pooled model is trained on.
"""

import logging

import polars as pl

logger = logging.getLogger(__name__)

TARGET_DERIVED_PATTERNS = [
    "Release_No",
    "^.*HOMA-IR.*$",
    "^.*FASTING_INSULIN.*$",
    "^.*DIABETES.*$",
]
"""Columns removed before training.

The insulin-resistance label is defined as ``HOMA-IR > 2.5`` and HOMA-IR is
computed from fasting insulin, so both -- and every interaction term derived from
them -- would leak the target. The interaction generator deliberately does not
exclude them (see :mod:`src.features.interactions`); this drop is what keeps them
out of the feature matrix.
"""

BASELINE_SELECTION = [
    "AGE",
    "BMI",
    "FASTING_GLUCOSE",
    "HBA1C",
    "HDL_C",
    "IR",
    "RACE",
    "SEX",
    "TG",
    "T_CHO",
]
"""The nine baseline variables plus the target, in selection order.

Note:
    The list is alphabetical, which puts ``IR`` in the middle rather than at the
    end. That order survives into the feature matrix once the target is split
    off, and column order decides how tree ensembles break ties between equally
    good splits, so it is fixed here rather than left to the caller.
"""

INTERACTION_PATTERN = "^.*[mul|log|div|sqrt].*$"
"""Pattern matching every generated interaction or transform column.

Note:
    Written as a character class, so it matches any column whose name contains
    any of the letters in ``mul|logdivsqrt`` -- far more than the intended four
    suffixes. It happens to select the right columns for this schema, and is
    reproduced verbatim rather than corrected, because narrowing it could change
    which columns reach the model.
"""


def drop_target_derived(df: pl.DataFrame) -> pl.DataFrame:
    """Remove the identifier and every column that leaks the target.

    Args:
        df: Feature table from :func:`src.features.interactions.generate_interactions`.

    Returns:
        The table without ``Release_No`` or any column derived from ``HOMA-IR``,
        ``FASTING_INSULIN`` or ``DIABETES``.
    """
    return df.drop(TARGET_DERIVED_PATTERNS)


def ablation_feature_sets(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Build the four nested feature sets compared in the ablation.

    The sets are nested: baseline variables, then all measured variables, then
    each of those with its generated interaction terms.

    Args:
        df: Feature table, already passed through :func:`drop_target_derived`.

    Returns:
        Ordered mapping of set name to a frame containing that set plus the
        ``IR`` target column.
    """
    baseline = df.select(BASELINE_SELECTION)
    extended = df.select(pl.all().exclude(INTERACTION_PATTERN))
    extended_with_interactions = df.clone()
    # Sorted only so the generated pattern is stable between runs; alternation is
    # order independent, so this cannot change which columns are matched.
    interaction_only_columns = sorted(set(extended.columns).difference(baseline.columns))
    baseline_with_interactions = df.select(
        pl.all().exclude("^.*(" + "|".join(interaction_only_columns) + ").*$")
    )

    sets = {
        "baseline": baseline,
        "baseline + interactions": baseline_with_interactions,
        "extended": extended,
        "extended + interactions": extended_with_interactions,
    }
    for name, frame in sets.items():
        logger.info("%s: %d features", name, frame.width - 1)
    return sets
