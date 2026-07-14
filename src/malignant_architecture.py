"""
Final manuscript figure: Figure 4 (see FIGURE_MAP.md).
figure5.py - Malignant heterogeneity, communication and CNV (Fig 5).

Regenerates the 6-panel composite (a schematic, b heterogeneity, c program
landscape, d cell-cell communication, e CNV heatmap, f CNV scores) and the
results digest from saved analysis outputs.

UPSTREAM ANALYSIS (documented; not re-run here - see Methods):
  Malignant heterogeneity scoring; program-landscape clustering;
  LIANA cell-cell communication over the TME; expression-based CNV inference
  (lineage-confound caveat noted in Methods). Products under
  heterogeneity/, communication/, cnv/, trajectory/.

INPUTS (saved outputs):
  /data/paper_tumoronly/Figure5/panels/Fig5[a-f]_*.png
  /data/paper_tumoronly/Figure5/heterogeneity/malignant_heterogeneity.csv
  /data/paper_tumoronly/Figure5/communication/{liana_results,curated_axes}.csv
  /data/paper_tumoronly/Figure5/cnv/cnv_scores_fullres.csv
OUTPUTS:
  /data/paper_tumoronly/Figure5/Figure5_full.{png,pdf}
  /data/paper_tumoronly/Figure5/Figure5_results.txt
"""
import os, sys, io, contextlib
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
sys.path.insert(0, os.path.dirname(__file__))
from atlas_setup import mm

FIGDIR = "/data/paper_tumoronly/Figure5"
PAN    = f"{FIGDIR}/panels"


def composite():
    panels = {"a": f"{PAN}/Fig5a_schematic.png", "b": f"{PAN}/Fig5b_heterogeneity.png",
              "c": f"{PAN}/Fig5c_landscape.png", "d": f"{PAN}/Fig5d_communication.png",
              "e": f"{PAN}/Fig5e_cnv_heatmap.png", "f": f"{PAN}/Fig5f_cnv_scores.png"}
    missing = [k for k, v in panels.items() if not os.path.exists(v)]
    if missing: print("WARNING missing panels:", missing)

    fig = plt.figure(figsize=mm(183, 250))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.7, 1.0, 1.0, 1.0], hspace=0.12, wspace=0.06)
    layout = {"a": gs[0, :], "b": gs[1, 0], "c": gs[1, 1],
              "d": gs[2, :], "e": gs[3, 0], "f": gs[3, 1]}
    for lab, cell in layout.items():
        ax = fig.add_subplot(cell); ax.axis("off")
        if os.path.exists(panels[lab]): ax.imshow(mpimg.imread(panels[lab]))
        ax.text(-0.01, 1.02, lab, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="right")
    fig.savefig(f"{FIGDIR}/Figure5_full.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{FIGDIR}/Figure5_full.pdf", bbox_inches="tight")
    plt.close(fig)


def digest():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        def hdr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)
        hdr("FIGURE 5 - RESULTS DIGEST (heterogeneity / communication / CNV)")
        het = f"{FIGDIR}/heterogeneity/malignant_heterogeneity.csv"
        if os.path.exists(het):
            df = pd.read_csv(het)
            print("malignant_heterogeneity.csv shape:", df.shape)
            print(df.head(10).to_string(index=False))
        ax = f"{FIGDIR}/communication/curated_axes.csv"
        if os.path.exists(ax):
            hdr("Conserved communication axes")
            print(pd.read_csv(ax).to_string(index=False))
        cnv = f"{FIGDIR}/cnv/cnv_scores_fullres.csv"
        if os.path.exists(cnv):
            c = pd.read_csv(cnv, index_col=0)
            hdr("CNV scores")
            print("cnv_scores_fullres.csv shape:", c.shape)
    open(f"{FIGDIR}/Figure5_results.txt", "w").write(buf.getvalue())


def main():
    composite()
    digest()
    print("Figure 5 regenerated.")


if __name__ == "__main__":
    main()
