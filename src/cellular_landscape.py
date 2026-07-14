"""
Final manuscript figure: Figure 1 (see FIGURE_MAP.md).
figure2.py - Shared cellular landscape across cancers (Fig 2, tumour-only).
Regenerates the 5-panel composite (a schematic, b cell-type UMAP, c composition
per cancer, d marker dotplot, e cross-cancer heatmap) + results digest.
The tumour-vs-normal proportions panel is intentionally omitted (tumour-only).
"""
import os, sys, io, contextlib
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.image as mpimg
import matplotlib.patheffects as pe
sys.path.insert(0, os.path.dirname(__file__))
from atlas_setup import (CANCER_ORDER, CANCER_ABBR, CELLTYPE_ORDER, CELLTYPE_COLORS,
                         set_nature_style, mm, save_fig)

FIGDIR = "/data/paper/Figure2"
PAN    = f"{FIGDIR}/panels"


def render_umap_panel():
    """Panel b: cell-type UMAP at 600 dpi, bold black axis text."""
    import scanpy as sc
    set_nature_style()
    mpl.rcParams.update({"savefig.bbox": "standard", "font.weight": "medium"})
    adata = sc.read_h5ad("/data/figures/adata_500k_annotated.h5ad")
    clean = adata[adata.obs["cell_type"] != "Low-quality/Doublet"].copy()

    xy = clean.obsm["X_umap"]; rng = np.random.default_rng(0)
    ridx = rng.permutation(clean.n_obs)
    present = [c for c in CELLTYPE_ORDER if c in clean.obs["cell_type"].unique()]
    cvec = clean.obs["cell_type"].astype(str).values
    col = np.array([CELLTYPE_COLORS.get(c, "#888888") for c in cvec])

    fig, ax = plt.subplots(figsize=mm(115, 105))
    ax.scatter(xy[ridx, 0], xy[ridx, 1], c=col[ridx], s=0.4, lw=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=11, fontweight="bold", color="black")
    ax.set_ylabel("UMAP2", fontsize=11, fontweight="bold", color="black")
    for c in present:
        m_ = cvec == c; cx, cy = np.median(xy[m_, 0]), np.median(xy[m_, 1])
        ax.text(cx, cy, c, fontsize=5, fontweight="bold", ha="center", va="center",
                color="black", path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    ax.legend(handles=[plt.Line2D([0], [0], marker='o', ls='', ms=3.8,
              mfc=CELLTYPE_COLORS[c], mec='none', label=c) for c in present],
              loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
              fontsize=6, labelspacing=0.3)
    ax.set_title("Cell types", loc="left", fontsize=13, fontweight="bold", color="black")
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("black"); ax.spines[sp].set_linewidth(0.9); ax.spines[sp].set_visible(True)
    ax.tick_params(colors="black")
    fig.savefig(f"{PAN}/Fig2A_umap_celltype.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(f"{PAN}/Fig2A_umap_celltype.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def composite():
    """Tile 5 panels (a-e) into the Figure 2 composite (tumour-only)."""
    panels = {"a": f"{PAN}/Fig2a_schematic.png", "b": f"{PAN}/Fig2A_umap_celltype.png",
              "c": f"{PAN}/Fig2B_composition.png", "d": f"{PAN}/Fig2C_markers.png",
              "e": f"{PAN}/Fig2E_composition_heatmap.png"}
    missing = [k for k, v in panels.items() if not os.path.exists(v)]
    if missing: print("WARNING missing panels:", missing)

    fig = plt.figure(figsize=mm(183, 235))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.5, 1.25, 0.95, 1.2], hspace=0.10, wspace=0.04)
    axmap = {"a": fig.add_subplot(gs[0, :]), "b": fig.add_subplot(gs[1, 0]),
             "c": fig.add_subplot(gs[1, 1]), "d": fig.add_subplot(gs[2, :]),
             "e": fig.add_subplot(gs[3, :])}
    for lab, ax in axmap.items():
        ax.axis("off")
        if os.path.exists(panels[lab]): ax.imshow(mpimg.imread(panels[lab]))
        ax.text(-0.01, 1.02, lab, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="right")
    fig.savefig(f"{FIGDIR}/Figure2_full.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{FIGDIR}/Figure2_full.pdf", bbox_inches="tight")
    plt.close(fig)


def digest():
    import scanpy as sc
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        def hdr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)
        obs = sc.read_h5ad("/data/figures/adata_500k_annotated.h5ad").obs
        bio = obs[~obs["cell_type"].isin(["Low-quality/Doublet", "Erythroid"])]
        hdr("FIGURE 2 - RESULTS DIGEST (tumour-only)")
        print(f"Cells annotated: {obs.shape[0]:,} | biological cell types: {bio['cell_type'].nunique()}")
        vc = obs["cell_type"].value_counts()
        hdr("Cell-type totals (% of atlas)")
        for c in [x for x in CELLTYPE_ORDER if x in vc.index]:
            print(f"  {c:<26} {vc[c]:>8,} ({100*vc[c]/len(obs):5.2f}%)")
        hdr("Dominant cell type per cancer")
        comp = pd.crosstab(obs["cancer_type"], obs["cell_type"], normalize="index") * 100
        for c in [x for x in CANCER_ORDER if x in comp.index]:
            top = comp.loc[c].idxmax()
            print(f"  {CANCER_ABBR.get(c,c):<8} {top} ({comp.loc[c,top]:.1f}%)")
    open(f"{FIGDIR}/Figure2_results.txt", "w").write(buf.getvalue())


def main():
    os.makedirs(PAN, exist_ok=True)
    render_umap_panel()
    composite()
    digest()
    print("Figure 2 regenerated.")


if __name__ == "__main__":
    main()
