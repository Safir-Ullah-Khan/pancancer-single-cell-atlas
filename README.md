# A pan-cancer single-cell atlas of transferable malignant identity

Analysis code for a pan-cancer single-cell RNA-seq atlas of 4,828,616 cells
across 30 cancer types and 1,010 patient samples.

The pipeline covers atlas assembly, integration and annotation, malignant
transcriptional-program discovery, a single-cell tumour classifier,
cross-platform validation against independent TCGA patients, a malignant
transcriptome taxonomy, and all figure-generation scripts.

## Repository structure

```
.
├── README.md
├── FIGURE_MAP.md            # maps each script to its final manuscript figure
├── requirements.txt
├── run_all.py               # pipeline order / documentation
├── pipeline/                # data download, conversion, merge
└── src/                     # analysis and figure-generation scripts
```

See `FIGURE_MAP.md` for the script-to-figure mapping. Scripts are named by
content because figure numbering changed during revision; each script's
docstring states its final figure.

## Data availability

- Processed AnnData objects (annotated cell-type atlas, immune compartment,
  malignant cells, and supporting objects) are deposited on Zenodo:
  https://doi.org/10.5281/zenodo.21135969
- The atlas is assembled from 64 previously published accessions
  (GEO / ArrayExpress / GSA-BioProject); users should cite the original studies.
- The full merged atlas (~100 GB) is not tracked here and is reproducible from
  the source accessions using the `pipeline/` scripts.

## Analysis notes

- All expression-based analyses use a per-cancer balanced subset of 481,738
  cells (maximum 16,666 cells per cancer type). `src/subset_validation.py`
  confirms this subset preserves the cellular composition of the full atlas
  (mean absolute difference 0.15%, r = 0.9998 across canonical lineage markers).
- The classifier is a multinomial logistic-regression model (2,000 highly
  variable genes, standardized features, stratified 70/30 split), chosen for
  interpretability and cross-platform transferability.
- Scripts reference two output trees (`/data/paper/` and
  `/data/paper_tumoronly/`) reflecting the analysis history.

## Requirements

See `requirements.txt`. Analyses were run on Python 3.11 with scanpy 1.11 and
anndata 0.12.

```bash
pip install -r requirements.txt
```

## Contact

Dr. Safir Ullah Khan, Moores Cancer Center, University of California, San Diego.
