"""
Final manuscript figure: Figure 2 (see FIGURE_MAP.md).
figure3.py - Conserved immune ecosystem across cancers (Fig 3, tumour-only).
Redraws the neutral 3a schematic, then tiles the 5-panel composite
(a schematic, b immune UMAP, c T/NK focus, d myeloid focus, e cross-cancer
heatmap) + results digest. The tumour-vs-normal infiltration panel is
intentionally omitted (tumour-only framing).
"""
import os, sys, io, contextlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
sys.path.insert(0, os.path.dirname(__file__))
from atlas_setup import (CANCER_ORDER, CANCER_ABBR, IMMUNE_ORDER, IMMUNE_COLORS,
                         set_nature_style, mm, save_fig)

FIGDIR = "/data/paper/Figure3"
PAN    = f"{FIGDIR}/panels"


def render_schematic():
    """Panel a: neutral conserved-immune-ecosystem schematic (no tumour-vs-normal)."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=mm(183, 55))
    ax.set_xlim(0, 100); ax.set_ylim(0, 32); ax.axis("off")

    def box(x, y, w, h, fc, ec, title, sub="", tc="black", ty=None, sy=None, fs=6.0, sfs=4.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.1",
                     fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, ty or y + h * 0.62, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc)
        if sub:
            ax.text(x + w / 2, sy or y + h * 0.28, sub, ha="center", va="center", fontsize=sfs, color=tc)

    def arrow(x1, y1, x2, y2, c="#444"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=10, lw=1.0, color=c))

    # Stage 1: conserved immune ecosystem
    ax.add_patch(FancyBboxPatch((2, 5), 26, 22, boxstyle="round,pad=0.4,rounding_size=1.5",
                 fc="#f7f7f7", ec="#999", lw=0.8, ls="--"))
    ax.text(15, 24.5, "Conserved immune\necosystem", ha="center", va="center",
            fontsize=6.5, fontweight="bold")
    keys = ["CD4 naive", "CD8 cytotoxic", "Regulatory T cell", "NK cytotoxic", "B memory",
            "Monocyte", "C1Q+ TAM", "cDC", "Neutrophil", "Mast cell", "Plasma cell", "CD8 exhausted"]
    dotcols = [IMMUNE_COLORS.get(c, "#888888") for c in keys]
    gx, gy = np.meshgrid(np.linspace(5, 25, 6), np.linspace(8, 17, 2))
    for i, (xx, yy) in enumerate(zip(gx.ravel(), gy.ravel())):
        if i < len(dotcols):
            ax.add_patch(Circle((xx, yy), 1.1, fc=dotcols[i], ec="white", lw=0.4))
    ax.text(15, 6, "22 subtypes . 30 cancers", ha="center", fontsize=4.6, color="#555")
    arrow(28.5, 16, 35, 16)

    # Stage 2: three functional axes (descriptive)
    box(35, 20.5, 30, 7, "#eef7f1", "#238b45", "CD8 differentiation",
        sub="cytotoxic -> EM -> exhausted", tc="#1b5e3a", ty=24.7, sy=21.9)
    box(35, 12, 30, 7, "#eef3fb", "#4363D8", "CD4 regulatory / helper",
        sub="Treg . CXCL13+ Tfh", tc="#1b3a6b", ty=16.2, sy=13.4)
    box(35, 3.5, 30, 7, "#fef6ee", "#e6870d", "Myeloid maturation",
        sub="monocyte -> C1Q+ / SPP1+ TAM", tc="#9a5a1e", ty=7.7, sy=4.9)
    arrow(65, 16, 72, 16)

    # Stage 3: outcome
    box(72, 11, 26, 10, "#f2eef7", "#6a51a3", "Immunosuppressive\nTME states",
        sub="across 30 cancers", tc="#4a3270", ty=16.6, sy=12.9, fs=6.3, sfs=4.6)

    ax.set_title("Pan-cancer immune ecosystem", loc="left", fontweight="bold", fontsize=8)
    save_fig(fig, f"{PAN}/Fig3a_schematic"); plt.close(fig)


def composite():
    panels = {"a": f"{PAN}/Fig3a_schematic.png", "b": f"{PAN}/Fig3b_immune_umap.png",
              "c": f"{PAN}/Fig3c_TNK_umap.png", "d": f"{PAN}/Fig3d_myeloid_umap.png",
              "e": f"{PAN}/Fig3e_TME_heatmap.png"}
    missing = [k for k, v in panels.items() if not os.path.exists(v)]
    if missing: print("WARNING missing panels:", missing)

    fig = plt.figure(figsize=mm(183, 210))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.42, 1.15, 1.15], hspace=0.08, wspace=0.04)
    axmap = {"a": fig.add_subplot(gs[0, :]), "b": fig.add_subplot(gs[1, 0]),
             "c": fig.add_subplot(gs[1, 1]), "d": fig.add_subplot(gs[2, 0]),
             "e": fig.add_subplot(gs[2, 1])}
    for lab, ax in axmap.items():
        ax.axis("off")
        if os.path.exists(panels[lab]): ax.imshow(mpimg.imread(panels[lab]))
        ax.text(-0.01, 1.02, lab, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="right")
    fig.savefig(f"{FIGDIR}/Figure3_full.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{FIGDIR}/Figure3_full.pdf", bbox_inches="tight")
    plt.close(fig)


def digest():
    import scanpy as sc
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        def hdr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)
        immc = sc.read_h5ad("/data/figures/adata_immune_clean.h5ad")
        n = immc.n_obs
        hdr("FIGURE 3 - RESULTS DIGEST (tumour-only)")
        print(f"Immune cells: {n:,} | subtypes: {immc.obs['immune_subtype'].nunique()}")
        vc = immc.obs["immune_subtype"].value_counts()
        hdr("Immune subtype counts (% of immune)")
        for s in [x for x in IMMUNE_ORDER if x in vc.index]:
            print(f"  {s:<24} {vc[s]:>7,} ({100*vc[s]/n:5.2f}%)")
        TNK = ["CD4 naive", "CD4 T memory", "CD4 Tfh/CXCL13+", "Regulatory T cell", "T naive/resting",
               "CD8 cytotoxic", "CD8 effector-memory", "CD8 exhausted", "NK cytotoxic", "NK CD56bright/tissue"]
        MYE = ["Monocyte", "C1Q+ TAM", "SPP1+ TAM", "cDC", "pDC", "Neutrophil", "Mast cell"]
        hdr("Compartment sizes")
        print(f"  T/NK:    {immc.obs['immune_subtype'].isin(TNK).sum():,} "
              f"({100*immc.obs['immune_subtype'].isin(TNK).mean():.1f}%)")
        print(f"  Myeloid: {immc.obs['immune_subtype'].isin(MYE).sum():,} "
              f"({100*immc.obs['immune_subtype'].isin(MYE).mean():.1f}%)")
        hdr("Top cancer per immune subtype")
        M = pd.read_csv(f"{FIGDIR}/fig3_immune_composition_pct.csv", index_col=0)
        for s in [x for x in IMMUNE_ORDER if x in M.columns]:
            print(f"  {s:<24} max in {CANCER_ABBR.get(M[s].idxmax(), M[s].idxmax()):<8} ({M[s].max():.1f}%)")
    open(f"{FIGDIR}/Figure3_results.txt", "w").write(buf.getvalue())


def main():
    os.makedirs(PAN, exist_ok=True)
    render_schematic()
    composite()
    digest()
    print("Figure 3 regenerated.")


if __name__ == "__main__":
    main()
