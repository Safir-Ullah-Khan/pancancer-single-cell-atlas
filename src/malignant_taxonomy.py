"""
malignant_taxonomy.py - Pan-cancer malignant transcriptome taxonomy.

Hierarchically clusters malignant expression similarity across 30 cancers into
tissue-of-origin families (average linkage, k=13), and cross-checks the families
against independent TCGA patient cross-classification (from tcga_validation).

UPSTREAM DATA:
  Figure6/model/cancer_similarity.csv   (30x30 malignant pseudobulk similarity)
  Figure8_validation/tcga_confusion_pct.csv  (independent-patient confusion)
OUTPUTS (DATADIR): panel_taxonomy.png, panel_similarity_heatmap.png,
  panel_tcga_families.png, malignant_taxonomy_families.csv, malignant_taxonomy_full.{png,pdf}
Final manuscript figure: Figure 7 (see FIGURE_MAP.md)
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list, set_link_color_palette
from scipy.spatial.distance import squareform
sys.path.insert(0, os.path.dirname(__file__))
from atlas_setup import CANCER_ABBR, set_nature_style, mm, save_fig

DATADIR = "/data/paper_tumoronly/Figure8_validation"
SIM = "/data/paper_tumoronly/Figure6/model/cancer_similarity.csv"
K = 13
abbr = lambda n: CANCER_ABBR.get(n, n)


def _linkage():
    sim = pd.read_csv(SIM, index_col=0)
    S = (sim.values.astype(float) + sim.values.astype(float).T) / 2
    np.fill_diagonal(S, S.max()); D = S.max() - S; np.fill_diagonal(D, 0)
    cancers = list(sim.index)
    Z = linkage(squareform(D, checks=False), method="average")
    return Z, S, cancers


def panels():
    import matplotlib as mpl
    set_nature_style(); mpl.rcParams.update({"savefig.bbox": "standard", "font.weight": "medium"})
    Z, S, cancers = _linkage()

    # dendrogram (k=13 coloured families)
    palette = ["#c0392b","#e6870d","#2e86c1","#238b45","#8e44ad","#16a085","#d81b60",
               "#f39c12","#2980b9","#27ae60","#8e6a3f","#c0392b","#7f8c8d"]
    set_link_color_palette(palette[:K])
    fig, ax = plt.subplots(figsize=mm(120, 155))
    dendrogram(Z, labels=[abbr(c) for c in cancers], orientation="left",
               color_threshold=Z[-(K-1), 2], above_threshold_color="#bbbbbb", ax=ax)
    ax.set_xlabel("Malignant transcriptome distance", fontsize=10, fontweight="bold", color="black")
    ax.set_title("Pan-cancer malignant taxonomy", loc="left", fontsize=10, fontweight="bold", color="black")
    ax.spines[["top","right","bottom"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors="black")
    plt.setp(ax.get_yticklabels(), fontsize=8.5, fontweight="bold", color="black")
    save_fig(fig, f"{DATADIR}/panel_taxonomy"); plt.close(fig)

    fam = pd.DataFrame({"cancer": cancers, "abbr": [abbr(c) for c in cancers],
                        "family": fcluster(Z, K, criterion="maxclust")}).sort_values("family")
    fam.to_csv(f"{DATADIR}/malignant_taxonomy_families.csv", index=False)

    # similarity heatmap ordered by dendrogram
    order = leaves_list(Z); oc = [cancers[i] for i in order]
    Sord = pd.DataFrame(S, index=cancers, columns=cancers).loc[oc, oc]
    fig, ax = plt.subplots(figsize=mm(115, 115))
    im = ax.imshow(Sord.values, cmap="magma", aspect="equal")
    ax.set_xticks(range(len(oc))); ax.set_xticklabels([abbr(c) for c in oc], rotation=90, fontsize=7.5, fontweight="bold", color="black")
    ax.set_yticks(range(len(oc))); ax.set_yticklabels([abbr(c) for c in oc], fontsize=7.5, fontweight="bold", color="black")
    ax.set_title("Malignant transcriptome similarity", loc="left", fontsize=10, fontweight="bold", color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("similarity", fontsize=8, fontweight="bold"); cb.ax.tick_params(labelsize=7)
    save_fig(fig, f"{DATADIR}/panel_similarity_heatmap"); plt.close(fig)

    # TCGA cross-classification families (concordance)
    confp = pd.read_csv(f"{DATADIR}/tcga_confusion_pct.csv", index_col=0)
    codes = list(confp.index)
    C = (confp.values + confp.values.T) / 2; np.fill_diagonal(C, 0)
    Dc = C.max() - C; np.fill_diagonal(Dc, 0)
    Zc = linkage(squareform(Dc, checks=False), method="average"); ordc = [codes[i] for i in leaves_list(Zc)]
    Cord = pd.DataFrame(C, index=codes, columns=codes).loc[ordc, ordc]
    fig, ax = plt.subplots(figsize=mm(105, 105))
    im = ax.imshow(Cord.values, cmap="Blues", aspect="equal", vmin=0, vmax=np.percentile(C, 98))
    ax.set_xticks(range(len(ordc))); ax.set_xticklabels(ordc, rotation=90, fontsize=8, fontweight="bold", color="black")
    ax.set_yticks(range(len(ordc))); ax.set_yticklabels(ordc, fontsize=8, fontweight="bold", color="black")
    ax.set_title("Cross-classification in independent TCGA patients", loc="left", fontsize=9.5, fontweight="bold", color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("mutual confusion (%)", fontsize=8, fontweight="bold"); cb.ax.tick_params(labelsize=7)
    save_fig(fig, f"{DATADIR}/panel_tcga_families"); plt.close(fig)


def composite():
    p = {"a": "Fig8tax_schematic.png", "b": "panel_taxonomy.png",
         "c": "panel_similarity_heatmap.png", "d": "panel_tcga_families.png"}
    fig = plt.figure(figsize=mm(183, 215))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.30, 1.35, 1.15], hspace=0.15, wspace=0.10)
    layout = {"a": gs[0, :], "b": gs[1:3, 0], "c": gs[1, 1], "d": gs[2, 1]}
    for lab, cell in layout.items():
        ax = fig.add_subplot(cell); ax.axis("off")
        fp = f"{DATADIR}/{p[lab]}"
        if os.path.exists(fp):
            ax.imshow(mpimg.imread(fp))
        ax.text(-0.01, 1.02, lab, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")
    fig.savefig(f"{DATADIR}/malignant_taxonomy_full.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{DATADIR}/malignant_taxonomy_full.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    panels()
    composite()
    print("malignant_taxonomy figure regenerated.")


if __name__ == "__main__":
    main()
