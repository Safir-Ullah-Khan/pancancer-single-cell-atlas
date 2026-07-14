#!/usr/bin/env python3
"""
discriminative_gene_heatmap.py - Discriminative gene expression.
Final manuscript figure: Supplementary Figure 3 (see FIGURE_MAP.md).

Heatmap of the top-3 discriminative genes per cancer type (ranked by the
multinomial classifier coefficient), showing z-scored mean log-normalized
expression across all 30 cancers. Expression is computed on the same
log-normalized representation used for classifier training, producing a
block-diagonal structure that confirms classification is driven by
cancer-specific identity genes.

INPUTS:
  /data/paper_tumoronly/Figure6/model/classifier.pkl   (keys: 'clf', 'features')
  /data/paper_tumoronly/Figure4/malignant_tumour_cells.h5ad
OUTPUTS:
  /data/paper_tumoronly/Figure6/panel_lognorm_expression.csv
  /data/paper_tumoronly/FigureS_panels/discriminative_gene_heatmap_named.{png,pdf}
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt

CLASSIFIER = "/data/paper_tumoronly/Figure6/model/classifier.pkl"
MALIGNANT  = "/data/paper_tumoronly/Figure4/malignant_tumour_cells.h5ad"
OUTDIR     = "/data/paper_tumoronly/FigureS_panels"
FIG6DIR    = "/data/paper_tumoronly/Figure6"
TOP_N      = 3

try:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from atlas_setup import CANCER_ABBR
    abbr = lambda c: CANCER_ABBR.get(c, c)
except Exception:
    abbr = lambda c: c

mpl.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"]})


def build_expression_matrix():
    obj = joblib.load(CLASSIFIER)
    clf, feats = obj["clf"], np.array(obj["features"])
    mal = sc.read_h5ad(MALIGNANT)
    mal.X = mal.layers["counts"].copy() if "counts" in mal.layers else mal.X.copy()
    sc.pp.normalize_total(mal, target_sum=1e4)
    sc.pp.log1p(mal)
    panels = {c: list(feats[np.argsort(clf.coef_[i])[::-1]][:25])
              for i, c in enumerate(clf.classes_)}
    all_genes = []
    for c in clf.classes_:
        all_genes += panels[c]
    all_genes = [g for g in dict.fromkeys(all_genes) if g in mal.var_names]
    cancers = sorted(mal.obs["cancer_type"].unique())
    expr = pd.DataFrame(index=cancers, columns=all_genes, dtype=float)
    for c in cancers:
        m = (mal.obs["cancer_type"] == c).values
        sub = mal[m][:, all_genes].X
        sub = np.asarray(sub.todense()) if hasattr(sub, "todense") else np.asarray(sub)
        expr.loc[c] = sub.mean(axis=0)
    os.makedirs(FIG6DIR, exist_ok=True)
    expr.to_csv(os.path.join(FIG6DIR, "panel_lognorm_expression.csv"))
    return clf, feats, expr


def make_heatmap(clf, feats, expr):
    ordered_genes, gene_cancer = [], []
    for i, cancer in enumerate(clf.classes_):
        top = [g for g in feats[np.argsort(clf.coef_[i])[::-1]]
               if g in expr.columns][:TOP_N]
        for g in top:
            if g not in ordered_genes:
                ordered_genes.append(g); gene_cancer.append(cancer)
    cancer_order = list(clf.classes_)
    M = expr.loc[cancer_order, ordered_genes]
    Z = ((M - M.mean(axis=0)) / M.std(axis=0).replace(0, np.nan)).fillna(0)
    Zt = Z.T
    fig, ax = plt.subplots(figsize=(4.7, 9.1))
    im = ax.imshow(Zt.values, aspect="auto", cmap="RdBu_r",
                   vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax.set_yticks(range(len(ordered_genes)))
    ax.set_yticklabels(ordered_genes, fontsize=4.6)
    ax.set_xticks(range(len(cancer_order)))
    ax.set_xticklabels([abbr(c) for c in cancer_order], fontsize=5.5, rotation=90)
    ax.set_xlabel("Cancer type", fontsize=7, fontweight="bold")
    ax.set_title("Top-3 discriminative genes per cancer", loc="left",
                 fontsize=8, fontweight="bold")
    prev = None
    for idx, c in enumerate(gene_cancer):
        if c != prev and idx > 0:
            ax.axhline(idx - 0.5, color="white", lw=0.4)
        prev = c
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("z-scored log-normalized expression", fontsize=6, fontweight="bold")
    cb.ax.tick_params(labelsize=5)
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "discriminative_gene_heatmap_named")
    fig.savefig(out + ".png", dpi=400, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf  ({len(ordered_genes)} genes x {len(cancer_order)} cancers)")


def main():
    clf, feats, expr = build_expression_matrix()
    make_heatmap(clf, feats, expr)


if __name__ == "__main__":
    main()
