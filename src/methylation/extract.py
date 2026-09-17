"""Reading DNA methylation arrays into a probe-by-sample matrix.

Two steps. The first unpacks one text file per participant from the vendor
archive and keeps the probes that pass the detection filter; the second joins
those per-participant tables into a single matrix of beta values.

Only the second step can be run from this repository. Taiwan Biobank methylation
is restricted-access data and is not distributed here, so the per-participant
tables are where this pipeline begins; see the Data Availability Statement in
``README.md``.

Platform:
    Illumina Infinium MethylationEPIC, not the earlier 450K array. The manifest
    at :data:`ANNOTATION_SUBPATH` carries 866,895 probes -- 863,904 ``cg`` sites
    annotated on GRCh37/hg19, 2,932 ``ch`` non-CpG sites and 59 ``rs`` genotyping
    controls -- against roughly 485,000 for 450K.

Normalisation:
    None is performed here. The archive supplies one ``*_nor.txt`` export per
    participant, holding beta values the data provider has already normalised,
    and nothing in this module re-normalises, background-corrects or
    batch-adjusts them. Which method the provider used is not recoverable from
    the exports and must come from the documentation accompanying the data
    release.

Probe filtering:
    Four filters, applied across the two steps: ``cg`` probes only, autosomes
    only (:data:`EXCLUDED_CHROMOSOMES`), a per-reading detection p-value below
    :data:`DETECTION_P_VALUE`, and complete-case -- a probe is kept only when
    every participant has a usable reading for it. On this cohort that is
    866,895 -> 863,904 -> 844,316 -> 332,284 probes.

    **No cross-reactive or SNP-overlap filter is applied.** Published
    cross-reactive probe lists are not consulted, and the manifest's ``SNP_ID``,
    ``SNP_DISTANCE`` and ``SNP_MINORALLELEFREQUENCY`` columns are carried through
    but unused. Probes reported as differentially methylated are therefore
    candidates for follow-up rather than a filtered final set.
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
        **This function cannot be run from this repository.** The vendor archive
        it reads is restricted-access Taiwan Biobank data and is not distributed
        here, so the pipeline starts from the per-participant tables instead. The
        function documents the extraction step for researchers who have their own
        approved access.

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
        Concatenating all samples side by side and then dropping incomplete rows
        would need the full eligible matrix in memory at once, which on this data
        is 8 GB; the two-pass route produces the same matrix in 3 GB. Probes are
        aligned by identifier rather than by position, so the result does not
        depend on the manifest's row order.

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
