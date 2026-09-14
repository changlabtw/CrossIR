"""Reading DNA methylation arrays into a probe-by-sample matrix.

Reimplements `4_MET.ipynb` cells [1] and the assembly half of cell [3].

The pipeline has two steps. The first unpacks one text file per participant from
the vendor archive and keeps the probes that pass the detection filter; the
second joins those per-sample tables into a single matrix of beta values.

Only the second step can be run from this repository. The archive the first step
reads lived on an external volume that is no longer available, so the per-sample
tables are the earliest reachable input. Taiwan Biobank methylation is
restricted-access data and is not distributed here in any case; see the Data
Availability Statement in ``README.md``.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

from src.data.io import raw_root, repo_path

logger = logging.getLogger(__name__)

ANNOTATION_SUBPATH = "20241028/MET_annotation/Epic_annotation.tsv"
"""Illumina EPIC manifest, relative to the raw data root."""

EXCLUDED_CHROMOSOMES = ["X", "Y"]
"""Sex chromosomes, dropped so that group differences cannot reflect sex alone."""

DETECTION_P_VALUE = 0.001
"""Detection p-value above which a probe reading is discarded as unreliable."""


def annotation_path() -> Path:
    """Locate the EPIC manifest.

    Returns:
        Absolute path to the annotation TSV.
    """
    return raw_root() / ANNOTATION_SUBPATH


def autosomal_cg_probes(path: Path | None = None) -> pl.DataFrame:
    """List the CpG probes eligible for testing.

    Keeps probes on the autosomes whose identifier begins with ``cg``, which
    excludes the control and single-nucleotide probes the manifest also carries.

    Args:
        path: Annotation TSV. Defaults to :func:`annotation_path`.

    Returns:
        Single column ``TargetID``, sorted by probe identifier.
    """
    probes = (
        pl.scan_csv(path or annotation_path(), separator="\t", infer_schema_length=0)
        .filter(~pl.col("CHR").is_in(EXCLUDED_CHROMOSOMES))
        .filter(pl.col("Probe_ID").str.slice(0, 2) == "cg")
        .select(pl.col("Probe_ID").alias("TargetID"))
        .sort("TargetID")
        .collect()
    )
    logger.info("Eligible probes: %d", probes.height)
    return probes


def extract_samples(zip_path, met_ids: list[str], out_dir) -> None:
    """Unpack per-sample beta values from the vendor archive.

    For each participant, reads the normalised intensity table from the archive,
    drops readings whose detection p-value is at or above
    :data:`DETECTION_P_VALUE`, joins the manifest to attach gene names, and
    writes one parquet per participant.

    Warning:
        **This function cannot be run from this repository.** The archive it
        reads lived on an external volume that is no longer available, so stage
        08 starts from the per-sample tables it once produced. The function is
        kept because it is part of the published method and a reader needs to see
        what was done. The legacy version of this code also carried a bug that
        would have raised ``TypeError`` -- ``is_in`` subscripted rather than
        called (``docs/audit.md`` F8) -- which is corrected here.

    Args:
        zip_path: Vendor archive of per-sample ``*_nor.txt`` files.
        met_ids: Methylation identifiers to extract, one per participant.
        out_dir: Directory to write the per-sample parquet files into.
    """
    import zipfile

    annotation = (
        pl.scan_csv(annotation_path(), separator="\t", infer_schema_length=0)
        .filter(~pl.col("CHR").is_in(EXCLUDED_CHROMOSOMES))
        .select(pl.col("Probe_ID").alias("TargetID"), pl.col("UCSC_REFGENE_NAME"))
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        for met_id in met_ids:
            member = f"MET/{met_id}/{met_id}_nor.txt"
            with archive.open(member) as handle:
                sample = (
                    pl.scan_csv(handle, separator="\t", ignore_errors=True)
                    .with_columns(
                        pl.col("AVG_Beta").cast(pl.Float64, strict=False),
                        pl.col("Detection Pval").cast(pl.Float64, strict=False),
                        pl.lit(met_id).alias("MET_ID"),
                    )
                    .filter(pl.col("Detection Pval") < DETECTION_P_VALUE)
                )
                annotation.join(sample, on="TargetID").select(
                    ["TargetID", "UCSC_REFGENE_NAME", "AVG_Beta", "Detection Pval", "MET_ID"]
                ).collect().write_parquet(out_dir / f"{met_id}.parquet")

    logger.info("Extracted %d samples to %s", len(met_ids), out_dir)


def samples_root() -> Path:
    """Return the directory holding the per-sample methylation tables.

    Returns:
        Absolute path configured as ``methylation_samples_root``.
    """
    return repo_path("methylation_samples_root")


def assemble_beta_matrix(sample_dir: Path | None = None, probes: pl.DataFrame | None = None):
    """Join the per-sample tables into one probe-by-sample matrix.

    A probe is kept only when every participant has a usable reading for it, so
    that no test is run on a varying subset. That is what reduces the eligible
    probes to the tested set.

    Note:
        Assembled in two passes over the per-sample files -- the first
        intersecting the probe sets, the second filling a preallocated array.
        The legacy version concatenated all samples side by side and then dropped
        incomplete rows, which needs the full eligible matrix in memory at once;
        on this data that is 8 GB against 16 GB of RAM. The two-pass route
        produces the same matrix in 3 GB. It also aligns the probe column
        explicitly: the legacy code paired an unsorted probe column with
        per-sample frames it had sorted, which was only correct because the
        manifest happens to be sorted already.

    Args:
        sample_dir: Directory of per-sample parquet files. Defaults to
            :func:`samples_root`.
        probes: Eligible probes as returned by :func:`autosomal_cg_probes`.
            Computed if not supplied.

    Returns:
        Frame with ``TargetID`` followed by one ``Float64`` column per
        participant, named by methylation identifier, sorted by probe.
    """
    sample_dir = Path(sample_dir or samples_root())
    files = sorted(sample_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No per-sample methylation tables in {sample_dir}")

    eligible = (probes if probes is not None else autosomal_cg_probes())["TargetID"]

    complete: pl.Series | None = None
    for index, file in enumerate(files, start=1):
        present = (
            pl.read_parquet(file, columns=["TargetID", "AVG_Beta"])
            .drop_nulls()["TargetID"]
        )
        complete = present if complete is None else complete.filter(complete.is_in(present))
        if index % 200 == 0:
            logger.info("Probe intersection after %d/%d samples: %d", index, len(files), len(complete))

    kept = eligible.filter(eligible.is_in(complete))
    logger.info("Probes complete in all %d samples: %d of %d", len(files), len(kept), len(eligible))

    index_of = {probe: position for position, probe in enumerate(kept.to_list())}
    matrix = np.empty((len(kept), len(files)), dtype=np.float64)

    for column, file in enumerate(files):
        sample = pl.read_parquet(file, columns=["TargetID", "AVG_Beta"]).drop_nulls()
        sample = sample.filter(pl.col("TargetID").is_in(kept))
        positions = np.fromiter(
            (index_of[probe] for probe in sample["TargetID"].to_list()),
            dtype=np.int64,
            count=sample.height,
        )
        matrix[positions, column] = sample["AVG_Beta"].to_numpy()
        if (column + 1) % 200 == 0:
            logger.info("Filled %d/%d samples", column + 1, len(files))

    frame = pl.DataFrame(
        {"TargetID": kept, **{file.stem: matrix[:, column] for column, file in enumerate(files)}}
    )
    logger.info("Beta matrix: %d probes x %d samples", frame.height, frame.width - 1)
    return frame
