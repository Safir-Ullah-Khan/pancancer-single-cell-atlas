#!/usr/bin/env python3
"""
run_all.py - Pipeline orchestrator for the pan-cancer single-cell atlas.

Documents the intended execution order. Each figure script in src/ is
self-contained and can be run individually; edit the paths inside each script
to match your environment. See FIGURE_MAP.md for the script-to-figure mapping.
"""
STEPS = [
    # --- data assembly (heavy; run once) ---
    ("pipeline/cancerscem_downloader_final.py", "Download source data",        "-"),
    ("pipeline/convert_to_h5ad.py",             "Per-sample matrices -> H5AD",  "-"),
    ("pipeline/04_merge_h5ad.py",               "Merge into unified atlas",     "-"),

    # --- atlas overview + validation ---
    ("src/atlas_overview.py",                "Atlas overview (composition, UMAP, QC)", "Supplementary Fig 1"),
    ("src/subset_validation.py",             "Balanced-subset composition validation", "Supplementary Fig 2"),

    # --- cellular + immune landscape ---
    ("src/cellular_landscape.py",            "Shared cellular landscape (17 cell types)", "Figure 1"),
    ("src/immune_ecosystem.py",              "Conserved immune ecosystem (22 subtypes)",  "Figure 2"),

    # --- malignant biology ---
    ("src/malignant_programs.py",            "Malignant transcriptional programs",        "Figure 3"),
    ("src/malignant_architecture.py",        "Heterogeneity, communication, CNV",         "Figure 4"),

    # --- classifier + validation ---
    ("src/ml_classifier.py",                 "ML classification of tumour identity",      "Figure 5"),
    ("src/discriminative_gene_heatmap.py",   "Top-3 discriminative genes per cancer",     "Supplementary Fig 3"),
    ("src/tcga_validation.py",               "Cross-platform TCGA validation",            "Figure 6"),
    ("src/malignant_taxonomy.py",            "Pan-cancer malignant taxonomy",             "Figure 7"),

    # --- clinical + robustness ---
    ("src/program_survival.py",              "Malignant-program survival associations",   "Supplementary Fig 4"),
    ("src/reproducibility.py",               "Robustness + resource summary",             "Supplementary Fig 5"),
]

if __name__ == "__main__":
    print(f"{'Script':<44}{'Figure':<22}Description")
    print("-" * 100)
    for script, desc, fig in STEPS:
        print(f"{script:<44}{fig:<22}{desc}")
    print("\nEach script is self-contained; edit paths to your environment, "
          "then run individually, e.g.:  python src/cellular_landscape.py")
