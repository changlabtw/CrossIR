"""Path resolution and transformations shared by all three cohort readers.

The three cohorts (NHANES, KNHANES, Taiwan Biobank) arrive in different file
formats with different variable names, but they converge on one schema. The
steps that are literally identical across the legacy
``scripts/data_processor.py`` classes live here; anything cohort-specific stays
in the cohort module.
"""

from pathlib import Path

import polars as pl
import yaml

HOMAIR_CUTOFF = 2.5
"""Threshold on HOMA-IR above which a participant is labelled insulin resistant.

Kept at the thesis value. A sensitivity analysis over alternative cut-offs is
recorded as a deferred decision in ``docs/migration-plan.md`` (X3).
"""

_PATHS_CACHE: dict | None = None
_COHORTS_CACHE: dict | None = None


def repo_root() -> Path:
    """Return the repository root directory.

    Returns:
        Absolute path to the directory containing ``src/`` and ``configs/``.
    """
    return Path(__file__).resolve().parents[2]


def load_paths() -> dict:
    """Read and cache ``configs/paths.yaml``.

    Returns:
        Mapping of path keys to their configured values. Relative values are
        left as written; resolve them with :func:`repo_path`.
    """
    global _PATHS_CACHE
    if _PATHS_CACHE is None:
        with open(repo_root() / "configs" / "paths.yaml", encoding="utf-8") as handle:
            _PATHS_CACHE = yaml.safe_load(handle)
    return _PATHS_CACHE


def load_cohorts() -> dict:
    """Read and cache ``configs/cohorts.yaml``.

    Returns:
        Mapping describing the raw inputs of each cohort: the ordered KNHANES
        file list and the NHANES survey cycles.
    """
    global _COHORTS_CACHE
    if _COHORTS_CACHE is None:
        with open(repo_root() / "configs" / "cohorts.yaml", encoding="utf-8") as handle:
            _COHORTS_CACHE = yaml.safe_load(handle)
    return _COHORTS_CACHE


def repo_path(key: str) -> Path:
    """Resolve one key of ``configs/paths.yaml`` to an absolute path.

    Args:
        key: Key in ``configs/paths.yaml``, e.g. ``"processed_root"``.

    Returns:
        The configured path, made absolute against the repository root when it
        is relative.

    Raises:
        KeyError: If the key is not present in the configuration file.
    """
    value = Path(load_paths()[key])
    return value if value.is_absolute() else repo_root() / value


def raw_root() -> Path:
    """Return the root of the raw cohort data.

    This tree is read in place and is never written to by this pipeline.

    Returns:
        Absolute path to the raw data directory.
    """
    return repo_path("raw_data_root")


def processed_path(name: str) -> Path:
    """Build a path inside ``data/processed`` and create the directory.

    Args:
        name: File name, e.g. ``"NHANES_data.parquet"``.

    Returns:
        Absolute path to the file (which need not exist yet).
    """
    root = repo_path("processed_root")
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def output_path(name: str) -> Path:
    """Build a path inside ``data/output`` and create the directory.

    ``data/output`` holds the committed figures and result tables.

    Args:
        name: File name, e.g. ``"stats.xlsx"``.

    Returns:
        Absolute path to the file (which need not exist yet).
    """
    root = repo_path("output_root")
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def write_parquet(df: pl.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to parquet, creating the parent directory.

    Args:
        df: Frame to write.
        path: Destination file path.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def derive_map(df: pl.DataFrame) -> pl.DataFrame:
    """Replace the four seated blood-pressure readings with mean arterial pressure.

    The two diastolic and the two systolic readings are averaged, then combined
    as ``MAP = (2 * diastolic + systolic) / 3``. Every column whose name ends in
    ``PRESSURE`` is dropped afterwards, leaving ``MAP`` as the only blood
    pressure feature.

    Args:
        df: Frame containing ``SIT_1_DIASTOLIC_PRESSURE``,
            ``SIT_2_DIASTOLIC_PRESSURE``, ``SIT_1_SYSTOLIC_PRESSURE`` and
            ``SIT_2_SYSTOLIC_PRESSURE``.

    Returns:
        The frame with a ``MAP`` column and no ``*PRESSURE`` columns.
    """
    return (
        df.with_columns(
            (
                (pl.col("SIT_1_DIASTOLIC_PRESSURE") + pl.col("SIT_2_DIASTOLIC_PRESSURE")) / 2
            ).alias("DIASTOLIC_PRESSURE")
        )
        .with_columns(
            (
                (pl.col("SIT_1_SYSTOLIC_PRESSURE") + pl.col("SIT_2_SYSTOLIC_PRESSURE")) / 2
            ).alias("SYSTOLIC_PRESSURE")
        )
        .with_columns(
            (
                (2 * pl.col("DIASTOLIC_PRESSURE") + pl.col("SYSTOLIC_PRESSURE")) / 3
            ).alias("MAP")
        )
        .select(pl.all().exclude("^.*PRESSURE$"))
    )


def add_homair(df: pl.DataFrame, cutoff: float = HOMAIR_CUTOFF) -> pl.DataFrame:
    """Add the HOMA-IR index and the binary insulin-resistance label.

    ``HOMA-IR = fasting insulin * fasting glucose / 405``, with fasting insulin
    in uU/mL and fasting glucose in mg/dL.

    Args:
        df: Frame containing ``FASTING_INSULIN`` and ``FASTING_GLUCOSE``.
        cutoff: Threshold above which ``IR`` is ``True``.

    Returns:
        The frame with ``HOMA-IR`` (float) and ``IR`` (boolean) columns.
    """
    return df.with_columns(
        (pl.col("FASTING_INSULIN") * pl.col("FASTING_GLUCOSE") / 405).alias("HOMA-IR")
    ).with_columns((pl.col("HOMA-IR") > cutoff).alias("IR"))
