"""Taiwan Biobank cohort reader.

Reads the survey and measurement tables, renames the variables onto the shared
schema, and applies the cohort filters.

Taiwan Biobank is restricted-access data: the raw files are not distributed with
this repository and must be obtained through Academia Sinica. See the Data
Availability Statement in ``README.md``.
"""

import logging
from pathlib import Path

import polars as pl

from src.data.io import derive_map, raw_root

logger = logging.getLogger(__name__)

TWB_SUBDIR = "20241028"
"""Release directory of the Taiwan Biobank extract under the raw data root."""

SURVEY_FEATURES = ["AGE", "SEX"]
"""Questionnaire variables kept from the survey table."""

MEASURE_FEATURES = [
    "BODY_WAISTLINE",
    "BMI",
    "SIT_1_DIASTOLIC_PRESSURE",
    "SIT_1_SYSTOLIC_PRESSURE",
    "SIT_2_DIASTOLIC_PRESSURE",
    "SIT_2_SYSTOLIC_PRESSURE",
    "FASTING_GLUCOSE",
    "HBA1C",
    "HDL_C",
    "LDL_C",
    "T_CHO",
    "TG",
    "SGOT",
    "SGPT",
    "BUN",
    "CREATININE",
    "URIC_ACID",
    "FAST_TIME",
]
"""Physical measurement and blood chemistry variables kept from the measure table.

Taiwan Biobank does not measure fasting insulin, so ``HOMA-IR`` cannot be
computed for this cohort. It is used as an external validation set, scored by a
model trained on NHANES and KNHANES.
"""

MINIMUM_FASTING_MINUTES = 480
"""Minimum fasting duration (8 hours) required for a participant to be kept."""


def _read_baseline(path: Path, columns: list[str]) -> pl.DataFrame:
    """Read one Taiwan Biobank CSV, keeping baseline visits only.

    Args:
        path: CSV file to read.
        columns: Columns to retain.

    Returns:
        Baseline rows restricted to ``columns``.
    """
    return (
        pl.scan_csv(path, ignore_errors=True)
        .filter(pl.col("FOLLOW") == "Baseline")
        .select(columns)
        .collect()
    )


def _parse_fasting_minutes(df: pl.DataFrame) -> pl.DataFrame:
    """Convert the free-text fasting duration into whole minutes.

    ``FAST_TIME`` is recorded as a string holding an hour and a minute figure;
    both integers are extracted and combined as ``hours * 60 + minutes``.

    Args:
        df: Frame with a string ``FAST_TIME`` column.

    Returns:
        The frame with ``FAST_TIME`` as an integer number of minutes.
    """
    return df.with_columns(
        pl.col("FAST_TIME")
        .str.extract_all(r"(\d+)")
        .map_elements(lambda parts: int(parts[0]) * 60 + int(parts[1]), return_dtype=pl.Int64)
    )


def process_twb(root: Path | None = None) -> pl.DataFrame:
    """Build the analysis-ready Taiwan Biobank table.

    Joins the questionnaire, measurement and methylation-identifier tables on
    ``Release_No``, keeps self-reported non-diabetics, drops any row with a
    missing value, derives mean arterial pressure, and keeps only participants
    who fasted for at least eight hours.

    Args:
        root: Raw data root. Defaults to the configured ``raw_data_root``.

    Returns:
        Taiwan Biobank participants with the shared schema and a ``MET_ID``
        column linking to the methylation array data (empty string when the
        participant has no array sample).
    """
    base = Path(root or raw_root()) / TWB_SUBDIR

    met_contrast = _read_baseline(base / "lab_info.csv", ["Release_No", "MET_ID"])
    survey = _read_baseline(
        base / "release_list_survey.csv", ["Release_No", "DIABETES"] + SURVEY_FEATURES
    )
    measure = _read_baseline(
        base / "measure" / "release_list_measure.csv", ["Release_No"] + MEASURE_FEATURES
    ).sort("Release_No")
    logger.info(
        "TWB rows read: lab_info=%d survey=%d measure=%d",
        met_contrast.height,
        survey.height,
        measure.height,
    )

    merged = (
        survey.join(measure, on="Release_No")
        .join(met_contrast, on="Release_No", how="left")
        .with_columns(pl.col("MET_ID").fill_null(""))
    )
    logger.info("TWB rows after merge: %d", merged.height)

    processed = (
        merged.filter(pl.col("DIABETES") == 0)
        .drop("DIABETES")
        .drop_nulls()
        .pipe(_parse_fasting_minutes)
        .pipe(derive_map)
        .filter(pl.col("FAST_TIME") >= MINIMUM_FASTING_MINUTES)
        .drop("FAST_TIME")
        .drop_nulls()
        .sort("Release_No")
    )
    logger.info("TWB rows after filters and null removal: %d", processed.height)
    return processed
