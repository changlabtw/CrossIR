# Cross-ethnic machine learning prediction of insulin resistance

<!-- Replace XXXXXXX with the Zenodo record id once the DOI is minted. -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Analysis code for a gradient boosting model that predicts insulin resistance in non-diabetic adults
from measurements a routine blood panel already contains — no fasting insulin assay required. The
model is trained on pooled US (NHANES) and Korean (KNHANES) cohorts, interpreted with SHAP, and
applied to the Taiwan Biobank cohort, whose predicted groups are then compared by DNA methylation.

Insulin resistance is defined by **HOMA-IR > 2.5**, where
`HOMA-IR = fasting insulin (µU/mL) × fasting glucose (mg/dL) / 405`. Participants with diabetes are
excluded, so the task is detecting the pre-diabetic state in people not yet diagnosed.

## Study design

Non-diabetic adults from NHANES and KNHANES, both with measured HOMA-IR, train and test the
classifiers. Transfer between the two cohorts is assessed in both directions before they are pooled.
The model trained on the pooled set is then applied to a Taiwan Biobank cohort that has no
fasting-insulin measurement and therefore no observed label, and the subset of those participants with
methylation array data is compared by predicted IR status.

```mermaid
flowchart TB
    NHANES["NHANES<br/>Non-diabetic adults<br/>n = 11,660"]
    KNHANES["KNHANES<br/>Non-diabetic adults<br/>n = 15,138"]
    POOLED["Pooled training set<br/>n = 26,798<br/>Gradient-boosting classifiers<br/>IR defined as HOMA-IR &gt; 2.5"]
    TWB["Taiwan Biobank<br/>n = 92,734, no fasting-insulin measurement<br/>IR status predicted by the final model"]
    MET["Methylation subset<br/>n = 1,199 with EPIC array data<br/>Compared by predicted IR status"]

    NHANES -.-|bidirectional transfer| KNHANES
    NHANES --> POOLED
    KNHANES --> POOLED
    POOLED --> TWB
    TWB --> MET

    classDef cohort fill:#DEEBF7,stroke:#0072B2,stroke-width:1.5px,color:#14213D
    classDef pooled fill:#FDF1DC,stroke:#E69F00,stroke-width:1.5px,color:#14213D
    classDef external fill:#DCF1EA,stroke:#009E73,stroke-width:1.5px,color:#14213D
    classDef omics fill:#F7E6F0,stroke:#CC79A7,stroke-width:1.5px,color:#14213D

    class NHANES,KNHANES cohort
    class POOLED pooled
    class TWB external
    class MET omics
```

The same figure is available as TikZ in [`output/study_design.tex`](output/study_design.tex) for the
manuscript.

## Results

| Cohort | Raw records | Analysed | Insulin resistant |
|---|---|---|---|
| NHANES 1999–2012 (US) | 71,916 | 11,660 | 5,183 (44.5%) |
| KNHANES 2019–2021 (Korea) | 22,559 | 15,138 | 4,226 (27.9%) |
| Pooled training set | — | 26,798 | 9,409 (35.1%) |
| Taiwan Biobank (external) | 196,963 | 92,734 | 19,596 (21.1%), **predicted** |

Taiwan Biobank measures no fasting insulin, so it carries no observed label and no accuracy can be
computed on it. It is used to ask whether the model's predictions behave like real labels.

**Pooled model**, 30% hold-out of 26,798 participants:

| Model | Features | AUC | Accuracy | Sensitivity | Specificity | NPV | PR-AUC |
|---|---|---|---|---|---|---|---|
| CatBoost | 241 | 0.880 | 0.799 | 0.778 | 0.810 | 0.871 | 0.816 |
| Soft voting (3 models) | 241 | 0.880 | 0.799 | 0.782 | 0.809 | 0.873 | 0.817 |
| **CatBoost, top 20 by SHAP** | **20** | **0.880** | **0.799** | **0.785** | **0.807** | **0.874** | 0.815 |

Reducing 241 engineered features to the twenty most influential costs nothing measurable, which is the
practical result: the deployable model needs twenty inputs.

**Cross-cohort transfer**, each model scored on the entire cohort it never saw: NHANES → KNHANES
AUC 0.862, KNHANES → NHANES AUC 0.860. Performance survives the ethnic transfer with a modest
penalty against the within-cohort ceiling of 0.868 and 0.878.

**Differential methylation.** The 1,199 Taiwan Biobank participants with array data were profiled on
the **Illumina Infinium MethylationEPIC** BeadChip, not the earlier 450K array: its manifest carries
866,895 probes, of which the 863,904 `cg` sites are annotated on GRCh37/hg19 — the assembly the
downstream gene-structure lookups use. Probes are restricted to those `cg` sites — dropping the 2,932
`ch` non-CpG and 59 `rs` control probes — then to the autosomes, so that a group difference cannot
simply reflect the sex imbalance between the groups. A reading is kept only when its detection p-value
is below 0.001, and a probe only when every participant has a usable reading for it, so that no test
runs on a varying subset. That cascade is 866,895 → 863,904 → 844,316 → **332,284** probes tested.

Of those, 22 differ between the predicted groups at q < 0.05 and |log2FC| > 0.2, spanning 12 named
genes including COL25A1 (`cg22266749`), PSMA6 (`cg02987832`) and ERV3-1 (`cg06513015`). The manifest
annotates those probes with 13 symbols, but UBE2QP1 is an alias of UBE2Q2P1, the official symbol.
Two caveats bound how far they should be read: the grouping variable is a model prediction rather
than a measurement, and **no cross-reactive or SNP-overlap probe filter is applied** — the
manifest's `SNP_ID`, `SNP_DISTANCE` and `SNP_MINORALLELEFREQUENCY` columns are carried but unused,
and published cross-reactive probe lists are not consulted. These 22 probes are therefore
hypothesis-generating candidates for follow-up, not a filtered final set.

## Repository layout

```
configs/      Tuned hyperparameters, cohort file inventory, probe annotation, paths
data/
  raw/        Raw cohort files (not distributed — see Data availability)
  processed/  Intermediate tables (not distributed)
output/       Result tables and figures (committed)
models/       Final CatBoost models and manifest.csv
notebooks/    01–08, one per step of the analysis
src/
  data/       Cohort readers: nhanes.py, knhanes.py, twb.py, io.py
  features/   Interaction generation and feature selection
  models/     Training, evaluation, SHAP, regression
  methylation/Extraction, differential testing, annotation, enrichment
  viz/        Figures and tables
```

## Setup

Requires **Python 3.12.8**. The pinned versions in `requirements.txt` are part of the reproduction
contract rather than a lower bound: the boosting libraries are seeded only by their own defaults, so
the reported metrics are guaranteed reachable under this exact set and not necessarily under another.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the raw cohort files under `data/raw/` following the layout in `configs/cohorts.yaml`. To read
them from somewhere else, create `configs/paths.local.yaml` overriding only the keys you need:

```yaml
raw_data_root: /path/to/your/data/raw
```

## Reproducing the analysis

Run the notebooks in order. Each reads from `data/processed/` and writes to `output/`, so a run
leaves every table and figure in the repository regenerated.

| # | Notebook | Produces |
|---|---|---|
| 01 | `01_data_preparation.ipynb` | Cleaned cohort tables, HOMA-IR label, 241 engineered features |
| 02 | `02_descriptive_statistics.ipynb` | `stats.xlsx` and `stats_original.xlsx`, boxplots, correlation matrix, HOMA-IR distributions |
| 03 | `03_homair_regression.ipynb` | Continuous HOMA-IR regression across six models |
| 04 | `04_feature_ablation.ipynb` | 9 / 17 / 57 / 241 feature sets compared |
| 05 | `05_cross_ethnic.ipynb` | Within-cohort hold-out and cross-cohort transfer |
| 06 | `06_final_model_and_shap.ipynb` | Pooled model, SHAP (default and colour-blind palettes), top-20 retrain, `models/` |
| 07 | `07_twb_validation.ipynb` | Taiwan Biobank scoring, TG/HDL-C comparison, Q-Q plot |
| 08 | `08_differential_methylation.ipynb` | Volcano plot, `met-point.xlsx`, `candidate.csv`, enrichment |

Two ordering constraints: **02 must run before 07**, which appends a fourth sheet to the workbooks 02
creates; and **07 must run before 08**, which splits participants by the label 07 predicts.
Notebooks 02–05 depend only on 01 and can otherwise run in any order.

Notebook 08 takes about twelve minutes, most of it 1.3 million statistical tests; everything else
runs in seconds to a few minutes.

**Figure colours.** Five figures use colour to carry meaning in a way that fails for the common forms
of colour vision deficiency, and each has a `_color_blind.png` sibling drawn from the
[Okabe–Ito palette](src/viz/palettes.py): `correlation_matrix` (a red-to-blue diverging scale),
`volcano_plot_of_DNA_methylation_analysis` (red points against a green threshold line), `ORA` (a
categorical palette pairing red with green), and `HOMA-IR` and `qqplot_NHANES+KNHANES_vs_TWB` (red
reference lines). `SHAP_summary_plot_new` has one for the same reason. The remaining figures already
encode safely and have no sibling: `ConfusionMatrix_CatBoost` is a single-hue sequential map,
`ROC_PR_CatBoost` is orange on blue, and the three boxplots separate IR− from IR+ with blue and
orange. In every pair the data, axes and ordering are identical — only the palette differs.

## Data availability

| Cohort | Access |
|---|---|
| **NHANES** (1999–2012) | Public. US CDC / National Center for Health Statistics: https://wwwn.cdc.gov/nchs/nhanes/ |
| **KNHANES** (2019–2021) | Public, subject to the agency's terms of use. Korea Disease Control and Prevention Agency: https://knhanes.kdca.go.kr/ |
| **Taiwan Biobank** | **Restricted.** Available to qualified researchers by application to Academia Sinica: https://www.twbiobank.org.tw/ |

This study's use of Taiwan Biobank data was approved under **TWBR11308-07**, and the research protocol
under IRB approval **NCCU-REC-201904-I017**. No individual-level data from any cohort is distributed
with this repository; `data/raw/` and `data/processed/` are excluded from version control.

**The methylation analysis cannot be re-run from its raw input here.** Notebook 08 begins from
per-participant methylation tables rather than from the vendor array archive, which is
restricted-access Taiwan Biobank data and is not distributed with this repository. The extraction step
is kept in `src/methylation/extract.py` as the specification of how those tables were produced, for
researchers with their own approved Taiwan Biobank methylation access.

Those tables hold **already-normalised** beta values: the archive supplies one `*_nor.txt` export per
participant with `AVG_Beta` and `Detection Pval` columns, and this pipeline applies no normalisation,
background correction or batch adjustment of its own. Which normalisation the data provider ran is
therefore outside this repository — anyone reproducing or extending the analysis should take that
detail from the Taiwan Biobank methylation documentation accompanying their own data release, since it
is not recoverable from the exports themselves.

## Citation

If you use this software or its results, please cite the accompanying paper and this repository. See
`CITATION.cff` for machine-readable metadata; fill in the DOI once Zenodo has minted it.

## License

MIT — see [LICENSE](LICENSE). The cohort data is **not** covered by this licence and remains subject to
each provider's own terms.
