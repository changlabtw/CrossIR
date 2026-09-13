"""KNHANES 2019-2021 cohort reader.

Reimplements ``KNHANESDataProcessor`` from the legacy project's
``scripts/data_processor.py`` as plain functions.
"""

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from src.data.io import add_homair, derive_map, load_cohorts, raw_root

logger = logging.getLogger(__name__)

COLUMN_MAPPING = {
    "ID": "Release_No",
    "DE1_dg": "DIABETES",
    "sex": "SEX",
    "age": "AGE",
    "HE_BMI": "BMI",
    "HE_dbp2": "SIT_1_DIASTOLIC_PRESSURE",
    "HE_sbp2": "SIT_1_SYSTOLIC_PRESSURE",
    "HE_dbp3": "SIT_2_DIASTOLIC_PRESSURE",
    "HE_sbp3": "SIT_2_SYSTOLIC_PRESSURE",
    "HE_TG": "TG",
    "HE_LDL_drct": "LDL_C",
    "HE_chol": "T_CHO",
    "HE_glu": "FASTING_GLUCOSE",
    "HE_insulin": "FASTING_INSULIN",
    "HE_HbA1c": "HBA1C",
    "HE_HDL_st2": "HDL_C",
    "HE_wc": "BODY_WAISTLINE",
    "HE_alt": "SGPT",
    "HE_ast": "SGOT",
    "HE_BUN": "BUN",
    "HE_crea": "CREATININE",
    "HE_Uacid": "URIC_ACID",
}
"""KNHANES variable codes mapped to the shared schema."""


def read_year(path: Path) -> pl.DataFrame:
    """Read and clean one KNHANES yearly file.

    Note:
        ``LDL_C`` is *recomputed* with the Friedewald equation
        ``T_CHO - HDL_C - TG / 5`` and overwrites the directly measured
        ``HE_LDL_drct`` value. Rows with a non-positive result are dropped. This
        is reproduced from the legacy code unchanged; NHANES and Taiwan Biobank
        keep their own LDL values, so the three cohorts do not share one LDL
        definition (``docs/migration-plan.md``, X2).

    Args:
        path: Path to one ``.sas7bdat`` file.

    Returns:
        Cleaned participants from that survey year, sorted on ``Release_No``.
    """
    frame = pd.read_sas(path, format="sas7bdat")
    logger.info("KNHANES %s rows read: %d", Path(path).name, frame.shape[0])

    return (
        pl.from_pandas(frame)
        .select(list(COLUMN_MAPPING))
        .rename(COLUMN_MAPPING)
        .with_columns(LDL_C=pl.col("T_CHO") - pl.col("HDL_C") - pl.col("TG") / 5)
        .filter(pl.col("LDL_C") > 0)
        .pipe(derive_map)
        .with_columns(pl.col("Release_No").cast(pl.Utf8))
        .filter(pl.col("AGE") > 18)
        .filter(pl.col("DIABETES") == 0.0)
        .pipe(add_homair)
        .drop_nulls()
        .sort("Release_No")
    )


def process_knhanes(root: Path | None = None, files: list[str] | None = None) -> pl.DataFrame:
    """Build the analysis-ready KNHANES table.

    Warning:
        The yearly files are concatenated in the order given by ``files``, and
        rows are sorted only *within* each year. The order therefore determines
        the row order of the result, which in turn determines the training and
        test membership produced downstream by
        ``train_test_split(..., random_state=30)``. The default order comes from
        ``configs/cohorts.yaml`` and is the one that produced the published
        results; it is not alphabetical. See ``docs/audit.md`` F22.

    Args:
        root: Raw data root. Defaults to the configured ``raw_data_root``.
        files: Ordered file names inside ``<root>/KNHANES``. Defaults to
            ``configs/cohorts.yaml``.

    Returns:
        KNHANES participants with the shared schema plus ``HOMA-IR`` and ``IR``.
    """
    root = Path(root or raw_root()) / "KNHANES"
    files = files or load_cohorts()["knhanes_files"]

    combined = pl.concat([read_year(root / name) for name in files])
    logger.info("KNHANES rows after filters and null removal: %d", combined.height)
    return combined
