"""Interaction feature generation.

Reimplements ``FeatureGenerator`` from the legacy project's
``scripts/feature_engineering.py`` as a single function.

For every eligible clinical variable the generator adds three unary transforms
(square, square root, base-10 logarithm) and, for every unordered pair, a product
and a ratio. Rows where any generated value is not finite are dropped.
"""

import logging

import polars as pl

logger = logging.getLogger(__name__)

EXCLUDED_FROM_INTERACTIONS = [
    "Release_No",
    "MET_ID",
    "IR",
    "SEX",
    "AGE",
    "RACE",
    "DIABETES",
]
"""Columns that are kept as features but never combined into interactions.

Identifiers and the target are excluded for obvious reasons. ``SEX``, ``AGE`` and
``RACE`` are excluded because products and ratios of a categorical code carry no
interpretable meaning.

Note:
    ``HOMA-IR`` and ``FASTING_INSULIN`` are *not* excluded here, so the generated
    table does contain interactions derived from the quantities the target is
    computed from. Every modelling stage drops those columns by pattern before
    training, which is what keeps the target out of the feature matrix. The
    exclusion list is reproduced from the legacy code unchanged.
"""


def generate_interactions(
    df: pl.DataFrame,
    race: int,
    exclude: list[str] | None = None,
) -> pl.DataFrame:
    """Add a cohort code, then all unary and pairwise interaction features.

    Columns are sorted alphabetically both before and after generation. The
    order matters downstream: the boosted-tree models sample columns by index
    (``colsample_bytree``, ``colsample_bylevel``), so renaming or reordering a
    column changes which features a tree sees.

    Args:
        df: Cleaned cohort table.
        race: Value written to the ``RACE`` column. This encodes the *source
            dataset*, not participant ethnicity: ``1`` for NHANES, ``2`` for
            KNHANES. Taiwan Biobank is scored as ``2`` because it is passed
            through the same code path as KNHANES.
        exclude: Columns not to build interactions from. Defaults to
            :data:`EXCLUDED_FROM_INTERACTIONS`.

    Returns:
        The input columns plus the generated features, alphabetically sorted,
        with every row containing a non-finite generated value removed.
    """
    exclude = EXCLUDED_FROM_INTERACTIONS if exclude is None else exclude

    df = df.with_columns(pl.lit(race).alias("RACE"))
    df = df.select(sorted(df.columns))

    feature_list = df.select(pl.all().exclude(exclude)).columns
    logger.info("Generating interactions from %d base features", len(feature_list))

    new_features = []
    for index, first in enumerate(feature_list):
        new_features.extend(
            [
                (pl.col(first) * pl.col(first)).alias(f"{first}_square"),
                pl.col(first).sqrt().alias(f"{first}_sqrt"),
                pl.when(pl.col(first) > 0)
                .then(pl.col(first).log10())
                .otherwise(None)
                .alias(f"{first}_log"),
            ]
        )
        for second in feature_list[index + 1 :]:
            new_features.extend(
                [
                    (pl.col(first) * pl.col(second)).alias(f"{first}_mul_{second}"),
                    pl.when(pl.col(second) != 0)
                    .then(pl.col(first) / pl.col(second))
                    .otherwise(None)
                    .alias(f"{first}_div_{second}"),
                ]
            )

    extended = df.with_columns(new_features)
    generated = [column for column in extended.columns if column not in df.columns]
    logger.info("Generated %d features; dropping rows with non-finite values", len(generated))

    for feature in generated:
        extended = extended.filter(pl.col(feature).is_finite())

    extended = extended.select(sorted(extended.columns))
    logger.info("Feature table: %d rows x %d columns", extended.height, extended.width)
    return extended
