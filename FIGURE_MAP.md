# Figure map

Scripts are named by content (not figure number), because figure numbering
changed during revision. This table maps each script to its final manuscript
figure. Each script's docstring also states its final figure.

| Script | Final manuscript figure | Description |
|--------|-------------------------|-------------|
| `atlas_overview.py` | **Supplementary Figure 1** | Atlas overview: composition, UMAP, per-cell QC |
| `subset_validation.py` | **Supplementary Figure 2** | Balanced-subset vs full-atlas composition validation |
| `cellular_landscape.py` | **Figure 1** | Shared cellular landscape (17 cell types) |
| `immune_ecosystem.py` | **Figure 2** | Conserved immune ecosystem (22 subtypes) |
| `malignant_programs.py` | **Figure 3** | Malignant transcriptional programs |
| `malignant_architecture.py` | **Figure 4** | Malignant heterogeneity, communication, CNV |
| `ml_classifier.py` | **Figure 5** | Machine-learning classification of tumour identity |
| `discriminative_gene_heatmap.py` | **Supplementary Figure 3** | Top-3 discriminative genes per cancer |
| `tcga_validation.py` | **Figure 6** | Cross-platform validation in independent TCGA patients |
| `malignant_taxonomy.py` | **Figure 7** | Pan-cancer malignant transcriptome taxonomy |
| `program_survival.py` | **Supplementary Figure 4** | Malignant-program survival associations (TCGA) |
| `reproducibility.py` | **Supplementary Figure 5** | Robustness (downsampling, split-half, concordance) + resource |

## Supporting modules
- `atlas_setup.py` — shared plotting style, cancer/cell-type/immune palettes,
  abbreviations, marker sets, and annotation helper functions. Imported by all
  figure scripts.
- `atlas_overview_panels.py` — panel-builder helpers for the atlas overview
  (cells-per-cancer, tumour/normal composition, per-cell QC).

## Directory note
Scripts reference two output trees reflecting the project's analysis history:
`/data/paper/` (early atlas figures) and `/data/paper_tumoronly/` (tumour-only
re-analysis). This is expected; paths are documented in each script's docstring.

## Supplementary tables
Supplementary tables (S1–S11) accompany the manuscript separately as one Excel
workbook per figure, numbered by order of first mention in Results.
