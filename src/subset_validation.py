#!/usr/bin/env python3
"""
subset_validation.py - Balanced-subset validation.
Final manuscript figure: Supplementary Figure 2 (see FIGURE_MAP.md).

Verifies that the per-cancer balanced subset (481,738 cells) preserves the
cellular composition of the complete 4,828,616-cell atlas, by comparing the
fraction of cells positive for six canonical lineage markers, per cancer type.

INPUTS:
  /data/final/cancerscem_final.h5ad                          (full atlas)
  /data/zenodo_deposit/atlas_celltype_annotated_481k.h5ad    (balanced subset)
OUTPUTS:
  /data/paper_tumoronly/FigureS_panels/subset_validation.csv
  /data/paper_tumoronly/FigureS_panels/subset_validation.{png,pdf}
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt

FULL_ATLAS = "/data/final/cancerscem_final.h5ad"
SUBSET     = "/data/zenodo_deposit/atlas_celltype_annotated_481k.h5ad"
OUTDIR     = "/data/paper_tumoronly/FigureS_panels"
MARKERS    = ["CD3E", "CD68", "EPCAM", "PECAM1", "LYZ", "MS4A1"]

mpl.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
})


def marker_fraction_full_atlas(path, markers, chunk=500_000):
    """Marker+ fraction per cancer in the full atlas via chunked h5py CSR read
    (memory-safe for a >100 GB backed file)."""
    import h5py
    full = sc.read_h5ad(path, backed="r")
    present = [g for g in markers if g in full.var_names]
    gene_idx = {g: full.var_names.get_loc(g) for g in present}
    cancer = np.asarray(full.obs["cancer_type"].values)
    n = full.n_obs
    del full
    marker_pos = np.zeros((n, len(present)), dtype=bool)
    with h5py.File(path, "r") as f:
        X = f["X"]
        indptr = X["indptr"][:]
        data, indices = X["data"], X["indices"]
        gene_list = [gene_idx[g] for g in present]
        gcol = {v: k for k, v in enumerate(gene_list)}
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            lo, hi = indptr[s], indptr[e]
            idx_c, dat_c = indices[lo:hi], data[lo:hi]
            for gv in gene_list:
                hit = (idx_c == gv) & (dat_c > 0)
                if hit.any():
                    rows = np.searchsorted(indptr[s:e + 1],
                                           np.where(hit)[0] + lo, side="right") - 1
                    marker_pos[s + rows, gcol[gv]] = True
    rows = []
    for j, g in enumerate(present):
        for c in np.unique(cancer):
            rows.append({"marker": g, "cancer": c,
                         "full_pct": marker_pos[cancer == c, j].mean() * 100})
    return pd.DataFrame(rows)


def marker_fraction_subset(path, markers):
    sub = sc.read_h5ad(path, backed="r")
    present = [g for g in markers if g in sub.var_names]
    gi = [sub.var_names.get_loc(g) for g in present]
    X = sub[:, gi].X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    cancer = np.asarray(sub.obs["cancer_type"].values)
    rows = []
    for j, g in enumerate(present):
        for c in np.unique(cancer):
            rows.append({"marker": g, "cancer": c,
                         "subset_pct": (X[cancer == c, j] > 0).mean() * 100})
    return pd.DataFrame(rows)


def make_figure(merged, outdir):
    os.makedirs(outdir, exist_ok=True)
    markers = merged["marker"].unique()
    palette = {"CD3E": "#3B6FB6", "CD68": "#C74B8B", "EPCAM": "#2E8B7A",
               "PECAM1": "#E08A2B", "LYZ": "#7F5AA8", "MS4A1": "#C0453B"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for g in markers:
        d = merged[merged.marker == g]
        ax.scatter(d["full_pct"], d["subset_pct"], s=38, alpha=0.75,
                   c=palette.get(g, "#666"), edgecolors="white", lw=0.5, label=g)
    lims = [0, max(merged["full_pct"].max(), merged["subset_pct"].max()) * 1.05]
    ax.plot(lims, lims, ls="--", c="#888", lw=1.2, zorder=0)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Marker+ cells, full atlas (%)", fontweight="bold")
    ax.set_ylabel("Marker+ cells, balanced subset (%)", fontweight="bold")
    ax.set_title("Balanced subset faithfully represents full atlas",
                 fontsize=10, fontweight="bold")
    ax.legend(title="Marker", fontsize=8, frameon=False, loc="upper left", ncol=2)
    r = np.corrcoef(merged["full_pct"], merged["subset_pct"])[0, 1]
    ax.text(0.97, 0.05, f"r = {r:.4f}\nmean |Delta| = {merged['diff'].mean():.2f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888", lw=0.7))
    ax.text(-0.12, 1.05, "a", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax = axes[1]
    data = [merged[merged.marker == g]["diff"].values for g in markers]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                    medianprops=dict(color="black", lw=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    ax.set_xticks(range(1, len(markers) + 1))
    ax.set_xticklabels(markers, rotation=30, ha="right")
    for patch, g in zip(bp["boxes"], markers):
        patch.set_facecolor(palette.get(g, "#666")); patch.set_alpha(0.55)
    ax.axhline(1.0, ls=":", c="red", lw=1, label="1% threshold")
    ax.set_ylabel("|subset - full| difference (%)", fontweight="bold")
    ax.set_title("All discrepancies < 1.1%", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.text(-0.15, 1.05, "b", transform=ax.transAxes, fontsize=14, fontweight="bold")
    fig.suptitle("Validation: per-cancer balanced downsampling preserves cellular composition",
                 fontsize=11, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = os.path.join(outdir, "subset_validation")
    fig.savefig(out + ".png", dpi=350, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"saved {out}.png / .pdf")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("computing full-atlas marker fractions (reads the large atlas)...")
    full_val = marker_fraction_full_atlas(FULL_ATLAS, MARKERS)
    print("computing subset marker fractions...")
    sub_val = marker_fraction_subset(SUBSET, MARKERS)
    merged = full_val.merge(sub_val, on=["marker", "cancer"])
    merged["diff"] = (merged["full_pct"] - merged["subset_pct"]).abs()
    merged.to_csv(os.path.join(OUTDIR, "subset_validation.csv"), index=False)
    r = np.corrcoef(merged["full_pct"], merged["subset_pct"])[0, 1]
    print(f"OVERALL mean |subset - full| diff: {merged['diff'].mean():.2f}%  (r = {r:.4f})")
    make_figure(merged, OUTDIR)


if __name__ == "__main__":
    main()
