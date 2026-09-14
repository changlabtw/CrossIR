"""Over-representation analysis of the differentially methylated genes.

Reimplements `5_analysis.ipynb` cell [45]. The genes carrying a significant probe
are tested against curated pathway and ontology libraries through Enrichr, to ask
whether they cluster in any biological process more than chance would predict.

Note:
    The legacy notebook also called ``gseapy.prerank`` (cell [48]) and named its
    figure ``ORA_GSEA.png``. That call was handed a single column of gene symbols
    with no ranking statistic, which is not a valid ranked list, and it returned
    an empty result -- **no GSEA was performed and none is reported in the
    thesis** (``docs/audit.md`` F38). It is not migrated, and the figure is named
    for what it actually shows.

Warning:
    Enrichr is a live web service whose libraries are versioned by year and
    updated over time, so this is the one step of the pipeline that can change
    without any code changing. Re-running it on 2026-09-14 reproduced the
    published result exactly -- the same nineteen terms in the same order, with
    identical p-values -- but a future re-run carries no such guarantee. The
    result table is written to ``data/output/ORA_result.xlsx`` so the figure can
    be rebuilt without querying again.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

GENE_SET_LIBRARIES = [
    "KEGG_2021_Human",
    "WikiPathways_2024_Human",
    "Reactome_Pathways_2024",
    "GO_Biological_Process_2025",
    "GO_Molecular_Function_2025",
    "GO_Cellular_Component_2025",
    "BioCarta_2016",
    "MSigDB_Hallmark_2020",
    "PPI_Hub_Proteins",
    "HumanCyc_2016",
]
"""Enrichr libraries queried, in the order the legacy notebook listed them."""

ALPHA = 0.05
"""Adjusted p-value below which a pathway is reported as enriched."""


def over_representation(
    genes: list[str],
    libraries: list[str] | None = None,
    alpha: float = ALPHA,
    organism: str = "human",
) -> pd.DataFrame:
    """Test a gene list for over-representation in curated pathway libraries.

    Args:
        genes: Gene symbols. Must be non-empty; probes in intergenic regions
            contribute no symbol and should be filtered out beforehand.
        libraries: Enrichr libraries to query. Defaults to
            :data:`GENE_SET_LIBRARIES`.
        alpha: Adjusted p-value threshold for a term to be kept.
        organism: Enrichr organism.

    Returns:
        The significant terms, sorted by combined score, descending, with the
        index reset. Columns are Enrichr's own, including ``Gene_set``,
        ``Term``, ``Overlap``, ``Adjusted P-value``, ``Combined Score`` and the
        overlapping ``Genes``.
    """
    import gseapy as gp

    results = gp.enrichr(
        gene_list=genes,
        gene_sets=libraries or GENE_SET_LIBRARIES,
        organism=organism,
        verbose=False,
    ).results

    significant = (
        results.loc[results["Adjusted P-value"] < alpha]
        .sort_values("Combined Score", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(
        "Enrichment: %d terms below adjusted p %s, from %d genes across %d libraries",
        len(significant),
        alpha,
        len(genes),
        len(libraries or GENE_SET_LIBRARIES),
    )
    return significant
