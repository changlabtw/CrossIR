"""Locating differentially methylated probes within gene structures.

Reimplements `5_analysis.ipynb` cells [35]-[39]. Each significant probe is looked
up against the Ensembl gene track of the hg19 assembly to decide whether it falls
inside a protein-coding sequence.

The legacy notebook queried the UCSC Genome Browser REST API live, at the point
of use. Here the query is a one-off whose result is committed as
``configs/probe_annotation.csv`` (decision D10), so the notebook runs offline and
the annotation cannot drift underneath a later re-run.
"""

import logging
from pathlib import Path

import polars as pl

from src.data.io import display_path, repo_root

logger = logging.getLogger(__name__)

UCSC_ENDPOINT = "https://api.genome.ucsc.edu/getData/track"
"""UCSC Genome Browser REST endpoint used by the one-off query."""

UCSC_GENOME = "hg19"
"""Assembly the EPIC manifest's coordinates refer to."""

UCSC_TRACK = "ensGene"
"""Ensembl gene track queried for coding-sequence boundaries."""

ANNOTATION_FILE = "configs/probe_annotation.csv"
"""Committed result of the one-off query, relative to the repository root."""

COMPLETE = "cmpl"
"""UCSC status meaning the coding sequence boundary is fully determined."""


def load_probe_annotation() -> pl.DataFrame:
    """Read the committed UCSC gene-track annotation.

    Returns:
        One row per probe-transcript pair: ``TargetID``, ``Position``, the
        transcript ``name``, its ``cdsStart``/``cdsEnd`` and the two status
        flags.

    Raises:
        FileNotFoundError: If the annotation has not been generated.
    """
    path = repo_root() / ANNOTATION_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{ANNOTATION_FILE} is missing; regenerate it with query_ucsc_regions()"
        )
    return pl.read_csv(path)


def query_ucsc_regions(probes: pl.DataFrame) -> pl.DataFrame:
    """Look up the transcripts overlapping each probe position.

    One request per probe, against a frozen assembly, so the result is stable.

    Warning:
        **One-off.** This is the script that produced
        ``configs/probe_annotation.csv``; the pipeline reads that file rather
        than calling this. Re-run it only to regenerate the annotation, and
        commit what it returns.

    Args:
        probes: Frame with ``TargetID``, ``CHR`` and ``MAPINFO`` (a 1-based
            genomic coordinate).

    Returns:
        The concatenated track rows, one per overlapping transcript, with
        ``TargetID`` and ``Position`` attached.
    """
    import requests

    rows = []
    for probe in probes.to_dicts():
        position = int(probe["MAPINFO"])
        response = requests.get(
            UCSC_ENDPOINT,
            params={
                "genome": UCSC_GENOME,
                "chrom": f"chr{probe['CHR']}",
                "start": position - 1,
                "end": position,
                "track": UCSC_TRACK,
            },
            timeout=30,
        )
        response.raise_for_status()
        track = pl.DataFrame(response.json()[UCSC_TRACK])
        identity = {"TargetID": [probe["TargetID"]], "Position": [position]}
        if track.is_empty():
            # No transcript covers this position. One placeholder row keeps the
            # probe in the table with every transcript field null, which is what
            # makes it fall through to "no_annotate" in classify_coding_region.
            rows.append(pl.DataFrame(identity))
        else:
            rows.append(
                track.with_columns(
                    pl.lit(probe["TargetID"]).alias("TargetID"),
                    pl.lit(position).alias("Position"),
                )
            )
        logger.info("%s: %d overlapping transcripts", probe["TargetID"], track.height)

    return pl.concat(rows, how="diagonal_relaxed")


def classify_coding_region(annotation: pl.DataFrame) -> pl.DataFrame:
    """Label each probe by the kind of region it falls in.

    A probe is ``coding`` when some transcript has both coding-sequence
    boundaries fully determined and the probe lies between them;
    ``non_coding`` when neither boundary is determined; and ``no_annotate`` when
    the track reports no boundaries at all.

    Note:
        These three conditions are not exhaustive. A probe whose transcripts all
        have exactly one determined boundary matches none of them and is left
        unlabelled -- three of the twenty-two significant probes in this study
        (``docs/audit.md`` F39). Reproduced as published rather than widened,
        since the labels feed a published table.

    Args:
        annotation: Output of :func:`query_ucsc_regions` or
            :func:`load_probe_annotation`.

    Returns:
        Two columns, ``TargetID`` and ``coding``, one row per labelled probe.
    """
    coding = (
        annotation.filter(pl.col("cdsStartStat") == COMPLETE)
        .filter(pl.col("cdsEndStat") == COMPLETE)
        .filter(pl.col("Position") >= pl.col("cdsStart"))
        .filter(pl.col("Position") <= pl.col("cdsEnd"))
        .select("TargetID")
        .unique()
        .with_columns(coding=pl.lit("coding"))
    )
    remaining = annotation.filter(~pl.col("TargetID").is_in(coding["TargetID"]))

    non_coding = (
        remaining.filter(pl.col("cdsStartStat") != COMPLETE)
        .filter(pl.col("cdsEndStat") != COMPLETE)
        .filter(pl.col("cdsStartStat").is_not_null())
        .filter(pl.col("cdsEndStat").is_not_null())
        .select("TargetID")
        .unique()
        .with_columns(coding=pl.lit("non_coding"))
    )
    unannotated = (
        remaining.filter(pl.col("cdsStartStat").is_null())
        .filter(pl.col("cdsEndStat").is_null())
        .select("TargetID")
        .unique()
        .with_columns(coding=pl.lit("no_annotate"))
    )

    labels = pl.concat([coding, non_coding, unannotated])
    logger.info(
        "Region labels: %d coding, %d non-coding, %d unannotated, %d unlabelled",
        coding.height,
        non_coding.height,
        unannotated.height,
        annotation["TargetID"].n_unique() - labels.height,
    )
    return labels


def write_probe_annotation(annotation: pl.DataFrame, path: Path | None = None) -> Path:
    """Save a query result as the committed annotation file.

    Args:
        annotation: Output of :func:`query_ucsc_regions`.
        path: Destination. Defaults to :data:`ANNOTATION_FILE` under the
            repository root.

    Returns:
        The path written to.
    """
    path = path or repo_root() / ANNOTATION_FILE
    annotation.write_csv(path)
    logger.info("Wrote %s (%d rows)", display_path(path), annotation.height)
    return path
