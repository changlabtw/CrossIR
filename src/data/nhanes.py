"""NHANES 1999-2012 cohort reader.

Reimplements ``NHANESDataProcessor`` from the legacy project's
``scripts/data_processor.py`` as plain functions. The filter set and the derived
columns are reproduced exactly; see ``docs/migration-plan.md`` for the
discrepancies (X1) that are deliberately carried over rather than corrected.
"""

import logging
import os
from functools import reduce
from pathlib import Path

import pandas as pd
import polars as pl

from src.data.io import add_homair, derive_map, load_cohorts, raw_root

logger = logging.getLogger(__name__)

COLUMN_MAPPING = {
    "SEQN": "Release_No",
    "DIQ010": "DIABETES",
    "RIAGENDR": "SEX",
    "RIDAGEYR": "AGE",
    "BMXBMI": "BMI",
    "BPXDI2": "SIT_1_DIASTOLIC_PRESSURE",
    "BPXSY2": "SIT_1_SYSTOLIC_PRESSURE",
    "BPXDI3": "SIT_2_DIASTOLIC_PRESSURE",
    "BPXSY3": "SIT_2_SYSTOLIC_PRESSURE",
    "LBXTR": "TG",
    "LBDLDL": "LDL_C",
    "LBXTC": "T_CHO",
    "LBXGLU": "FASTING_GLUCOSE",
    "LBXIN": "FASTING_INSULIN",
    "LBXGH": "HBA1C",
    "LBDHDD": "HDL_C",
    "BMXWAIST": "BODY_WAISTLINE",
    "LBXSATSI": "SGPT",
    "LBXSASSI": "SGOT",
    "LBXSBU": "BUN",
    "LBXSCR": "CREATININE",
    "LBXSUA": "URIC_ACID",
}
"""NHANES variable codes mapped to the shared schema."""

LEGACY_RENAME = {
    "LBXHDD": "LBDHDD",
    "LBDHDL": "LBDHDD",
    "LBDSCR": "LBXSCR",
    "LBDSTB": "LBXSTB",
}
"""Cycle-to-cycle variable renames.

NHANES changed the HDL cholesterol and creatinine variable codes between survey
cycles. These are normalised before column selection so that every cycle exposes
the same names.
"""


def read_cycle(year: int, root: Path | None = None) -> pd.DataFrame:
    """Read and merge every ``.xpt`` file of one NHANES survey cycle.

    All files in the cycle directory are outer-merged on ``SEQN``, so a
    participant is kept even when they are absent from some of the component
    files. Merge order does not affect the result: the row set is order
    independent and :func:`process_nhanes` sorts globally before writing.

    Args:
        year: Cycle start year, e.g. ``2007`` for the 2007-2008 cycle.
        root: Raw data root. Defaults to the configured ``raw_data_root``.

    Returns:
        One row per participant, restricted to the columns in
        :data:`COLUMN_MAPPING` (under their NHANES codes).
    """
    root = root or raw_root()
    cycle_dir = Path(root) / "NHANES" / str(year)
    frames = [
        pd.read_sas(cycle_dir / name, format="xport")
        for name in sorted(os.listdir(cycle_dir))
        if Path(name).suffix == ".xpt"
    ]
    merged = reduce(lambda left, right: pd.merge(left, right, on="SEQN", how="outer"), frames)
    return merged.rename(columns=LEGACY_RENAME).loc[:, list(COLUMN_MAPPING)]


def process_nhanes(root: Path | None = None, years: list[int] | None = None) -> pl.DataFrame:
    """Build the analysis-ready NHANES table.

    Applies, in order: the shared column names, ``AGE > 18``, retention of
    self-reported non-diabetics (``DIQ010`` in ``{2, 3}`` -- "no" and
    "borderline"), mean arterial pressure, a six-digit zero-padded identifier,
    HOMA-IR with its binary label, removal of every row containing a null, and a
    sort on the identifier.

    Note:
        ``AGE > 18`` excludes participants aged exactly 18, whereas the thesis
        Methods section describes the criterion as age 18 and over. No fasting
        duration filter is applied; fasting glucose and insulin are measured only
        in the morning fasting subsample. Both behaviours are reproduced as
        written in the legacy code (``docs/migration-plan.md``, X1).

    Args:
        root: Raw data root. Defaults to the configured ``raw_data_root``.
        years: Survey cycles to include. Defaults to ``configs/cohorts.yaml``.

    Returns:
        NHANES participants with the shared schema plus ``HOMA-IR`` and ``IR``.
    """
    years = years or load_cohorts()["nhanes_years"]

    frames = [read_cycle(year, root) for year in years]
    combined = pl.from_pandas(pd.concat(frames))
    logger.info("NHANES rows read: %d", combined.height)

    processed = (
        combined.rename(COLUMN_MAPPING)
        .filter(pl.col("AGE") > 18)
        .filter(pl.col("DIABETES").is_between(2.0, 3.0))
        .pipe(derive_map)
        .with_columns(pl.col("Release_No").cast(pl.Int64).cast(pl.Utf8).str.zfill(6))
        .pipe(add_homair)
        .drop_nulls()
        .sort("Release_No")
    )
    logger.info("NHANES rows after filters and null removal: %d", processed.height)
    return processed
