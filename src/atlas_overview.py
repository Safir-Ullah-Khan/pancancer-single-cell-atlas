"""
Final manuscript figure: Supplementary Figure 1 (see FIGURE_MAP.md).
figure1.py — Pan-cancer atlas overview (Fig 1).
Regenerates the 6-panel composite (a schematic, b UMAP by cancer, c split
tumour/normal UMAPs, d cells per cancer, e sample composition, f per-cell QC)
plus the results digest, from saved processed objects.

Inputs (on disk):
  /data/figures/adata_500k_annotated.h5ad   (UMAP, cancer_type, sample_type)
  /data/figures/fig1_counts_by_cancer.csv
  /data/figures/fig1_qc_percell.csv
  /data/paper/Figure1/panels/Fig1a_schematic.png
Outputs:
  /data/paper/Figure1/Figure1_full.{png,pdf}
  /data/paper/Figure1/Figure1_results.txt
"""
import os, sys, io, contextlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
sys.path.insert(0, os.path.dirname(__file__))
from atlas_setup import (CANCER_ORDER, CANCER_ABBR, CANCER_COLORS,
                         set_nature_style, mm, save_fig)

FIGDIR = "/data/paper/Figure1"
PAN    = f"{FIGDIR}/panels"
ST     = {"Tumour": "#B2182B", "Normal": "#2166AC"}
abbr   = lambda n: CANCER_ABBR.get(n, n)


def load():
    import scanpy as sc
    adata = sc.read_h5ad("/data/figures/adata_500k_annotated.h5ad")
    ct = pd.read_csv("/data/figures/fig1_counts_by_cancer.csv", index_col=0)
    qc = pd.read_csv("/data/figures/fig1_qc_percell.csv")
    return adata, ct, qc


def composite(adata, ct, qc):
    set_nature_style()
    xy = adata.obsm["X_umap"]; rng = np.random.default_rng(0)
    ridx = rng.permutation(adata.n_obs)

    fig = plt.figure(figsize=mm(183, 300))
    outer = fig.add_gridspec(5, 1, height_ratios=[0.95, 0.15, 0.85, 1.45, 1.55], hspace=0.5)

    # row 0: a schematic | b UMAP by cancer (equal width)
    gAB = outer[0].subgridspec(1, 2, wspace=0.15)
    axA = fig.add_subplot(gAB[0, 0]); axA.axis("off")
    if os.path.exists(f"{PAN}/Fig1a_schematic.png"):
        axA.imshow(mpimg.imread(f"{PAN}/Fig1a_schematic.png")); axA.set_anchor("N")
    axB = fig.add_subplot(gAB[0, 1])
    colB = np.array([CANCER_COLORS.get(c, "#888") for c in adata.obs["cancer_type"].astype(str)])
    axB.scatter(xy[ridx, 0], xy[ridx, 1], c=colB[ridx], s=0.35, lw=0, rasterized=True)
    axB.set_xticks([]); axB.set_yticks([]); axB.set_xlabel("UMAP1"); axB.set_ylabel("UMAP2")
    axB.set_title("UMAP by cancer type", loc="left", fontweight="bold")

    # row 1: cancer color legend
    axLeg = fig.add_subplot(outer[1]); axLeg.axis("off")
    cats = [c for c in CANCER_ORDER if c in adata.obs["cancer_type"].unique()]
    axLeg.legend(handles=[plt.Line2D([0], [0], marker='o', ls='', ms=3.5,
                 mfc=CANCER_COLORS[c], mec='none', label=abbr(c)) for c in cats],
                 loc="center", ncol=10, frameon=False, fontsize=5,
                 handletextpad=0.25, columnspacing=0.7, labelspacing=0.4)

    # row 2: c split tumour/normal UMAPs
    gC = outer[2].subgridspec(1, 2, wspace=0.1)
    st = adata.obs["sample_type"].astype(str).values; cax = []
    for j, grp in enumerate(["Tumour", "Normal"]):
        axc = fig.add_subplot(gC[0, j]); cax.append(axc); m = st == grp
        axc.scatter(xy[:, 0], xy[:, 1], c="#dddddd", s=0.25, lw=0, rasterized=True)
        axc.scatter(xy[m, 0], xy[m, 1], c=ST[grp], s=0.3, lw=0, rasterized=True)
        axc.set_xticks([]); axc.set_yticks([]); axc.set_xlabel("UMAP1")
        if j == 0: axc.set_ylabel("UMAP2")
        axc.set_title(f"{grp} (n={m.sum():,})", loc="left", fontweight="bold", color=ST[grp])

    # row 3: d cells per cancer | e sample composition
    order = ct.sort_values("total").index; y = np.arange(len(order))
    gD = outer[3].subgridspec(1, 2, wspace=0.25)
    axD1 = fig.add_subplot(gD[0, 0]); axD2 = fig.add_subplot(gD[0, 1])
    axD1.barh(y, ct.loc[order, "total"].values,
              color=[CANCER_COLORS.get(c, "#888") for c in order], height=0.8)
    axD1.set_yticks(y); axD1.set_yticklabels([abbr(c) for c in order], fontsize=5)
    axD1.set_ylim(-0.5, len(order) - 0.5); axD1.set_xscale("log"); axD1.set_xlabel("Cells (n)")
    axD1.set_xlim(left=max(1, ct["total"].min() * 0.6)); axD1.tick_params(axis="y", length=0)
    axD1.spines["left"].set_visible(False); axD1.set_title("Cells per cancer", loc="left", fontweight="bold")
    tum = ct.loc[order, "pct_tumour"].values
    axD2.barh(y, tum, color=ST["Tumour"], height=0.8, label="Tumour")
    axD2.barh(y, 100 - tum, left=tum, color=ST["Normal"], height=0.8, label="Normal")
    axD2.set_yticks(y); axD2.set_yticklabels([]); axD2.set_ylim(-0.5, len(order) - 0.5)
    axD2.tick_params(axis="y", length=0); axD2.set_xlim(0, 100); axD2.set_xlabel("Composition (%)")
    axD2.spines["left"].set_visible(False)
    axD2.legend(loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.12))
    axD2.set_title("Sample composition", loc="left", fontweight="bold")

    # row 4: f per-cell QC
    gE = outer[4].subgridspec(3, 1, hspace=0.22); axE = [fig.add_subplot(gE[i, 0]) for i in range(3)]
    metrics = [("n_genes", "Genes / cell", True), ("n_umi", "UMIs / cell", True), ("pct_mt", "% mito", False)]
    eorder = [c for c in CANCER_ORDER if c in qc["cancer_type"].unique()]; rng2 = np.random.default_rng(0)
    for ax, (col, label, logy) in zip(axE, metrics):
        data = []
        for c in eorder:
            v = qc.loc[qc["cancer_type"] == c, col].dropna().values
            if len(v) > 4000: v = rng2.choice(v, 4000, replace=False)
            data.append(v)
        bp = ax.boxplot(data, positions=np.arange(len(eorder)), widths=0.6, showfliers=False,
                        patch_artist=True, medianprops=dict(color="black", linewidth=0.6),
                        whiskerprops=dict(linewidth=0.4), capprops=dict(linewidth=0.4),
                        boxprops=dict(linewidth=0.4))
        for patch, c in zip(bp["boxes"], eorder):
            patch.set_facecolor(CANCER_COLORS.get(c, "#888")); patch.set_edgecolor("black")
        if logy: ax.set_yscale("log")
        ax.set_ylabel(label); ax.set_xlim(-0.6, len(eorder) - 0.4); ax.set_xticks(np.arange(len(eorder)))
        if ax is axE[-1]:
            ax.set_xticklabels([abbr(c) for c in eorder], rotation=90, fontsize=5)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    axE[0].set_title("Per-cell QC by cancer type", loc="left", fontweight="bold")

    fig.canvas.draw()
    for ax, lab in [(axA, "a"), (axB, "b"), (cax[0], "c"), (axD1, "d"), (axD2, "e"), (axE[0], "f")]:
        p = ax.get_position(); fig.text(max(0.004, p.x0 - 0.045), p.y1 + 0.006, lab,
                                        fontsize=10, fontweight="bold")
    save_fig(fig, f"{FIGDIR}/Figure1_full"); plt.close(fig)


def digest(ct, qc):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        def hdr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)
        hdr("FIGURE 1 — RESULTS DIGEST")
        total = int(ct["total"].sum())
        print(f"Atlas total cells: {total:,} | Cancer types: {ct.shape[0]}")
        print(f"Tumour fraction: {(ct['total']*ct['pct_tumour']/100).sum()/total*100:.1f}%")
        s = ct.sort_values("total", ascending=False)
        hdr("Cells per cancer (largest / smallest)")
        for c in s.index[:5]: print(f"  {CANCER_ABBR.get(c,c):<8} {int(s.loc[c,'total']):>9,}")
        for c in s.index[-5:]: print(f"  {CANCER_ABBR.get(c,c):<8} {int(s.loc[c,'total']):>9,}")
        print(f"  Median: {int(ct['total'].median()):,}")
        hdr("Per-cell QC (median)")
        for col, lab in [("n_genes","Genes/cell"),("n_umi","UMIs/cell"),("pct_mt","% mito")]:
            v = qc[col].dropna(); print(f"  {lab:<12} {v.median():.0f} (IQR {v.quantile(.25):.0f}-{v.quantile(.75):.0f})")
    open(f"{FIGDIR}/Figure1_results.txt", "w").write(buf.getvalue())


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    adata, ct, qc = load()
    composite(adata, ct, qc)
    digest(ct, qc)
    print("Figure 1 regenerated.")


if __name__ == "__main__":
    main()
