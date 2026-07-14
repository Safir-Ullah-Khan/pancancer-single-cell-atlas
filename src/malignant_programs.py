"""
Final manuscript figure: Figure 3 (see FIGURE_MAP.md).
figure4.py - Malignant transcriptional programs (Fig 4).

Regenerates the 6-panel composite (a schematic, b state heatmap, c programs,
d violins, e state co-occurrence, f TF drivers) and the results digest from
saved analysis outputs.

UPSTREAM ANALYSIS (documented; not re-run here - see Methods):
  Malignant cells (89,530) isolated from tumour specimens ->
  NMF + curated program scoring (scanpy score_genes) ->
  program recurrence / co-occurrence across cancers ->
  lightweight TF-driver correlation (SCENIC not tractable in-env).
  Products: programs/*.csv, state_scores/*.csv, malignant_tumour_cells.h5ad

INPUTS (saved outputs):
  /data/paper_tumoronly/Figure4/panels/Fig4[a-f]_*.png
  /data/paper_tumoronly/Figure4/state_scores/state_scores_by_cancer.csv
  /data/paper_tumoronly/Figure4/programs/recurrent_programs_summary.csv
  /data/paper_tumoronly/Figure4/malignant_tumour_cells.h5ad
OUTPUTS:
  /data/paper_tumoronly/Figure4/Figure4_full.{png,pdf}
  /data/paper_tumoronly/Figure4/Figure4_results.txt
"""
import os, sys, io, contextlib
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
sys.path.insert(0, os.path.dirname(__file__))
from atlas_setup import mm

FIGDIR = "/data/paper_tumoronly/Figure4"
PAN    = f"{FIGDIR}/panels"


def composite():
    panels = {"a": f"{PAN}/Fig4a_schematic.png", "b": f"{PAN}/Fig4b_state_heatmap.png",
              "c": f"{PAN}/Fig4c_programs.png",  "d": f"{PAN}/Fig4d_violins.png",
              "e": f"{PAN}/Fig4e_cooccurrence.png", "f": f"{PAN}/Fig4f_tfdrivers.png"}
    missing = [k for k, v in panels.items() if not os.path.exists(v)]
    if missing: print("WARNING missing panels:", missing)

    fig = plt.figure(figsize=mm(183, 230))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.0, 1.0], hspace=0.12, wspace=0.06)
    axmap = {"a": fig.add_subplot(gs[0, :]), "b": fig.add_subplot(gs[1, 0]),
             "c": fig.add_subplot(gs[1, 1]), "d": fig.add_subplot(gs[2, 0]),
             "e": fig.add_subplot(gs[2, 1]), "f": None}
    # a spans top; b/c middle; d/e bottom; f appended below if present
    # simpler robust 3x2 with f in place of a second bottom cell:
    fig.clf()
    fig = plt.figure(figsize=mm(183, 250))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.7, 1.0, 1.0, 1.0], hspace=0.12, wspace=0.06)
    layout = {"a": gs[0, :], "b": gs[1, 0], "c": gs[1, 1],
              "d": gs[2, 0], "e": gs[2, 1], "f": gs[3, :]}
    for lab, cell in layout.items():
        ax = fig.add_subplot(cell); ax.axis("off")
        if os.path.exists(panels[lab]): ax.imshow(mpimg.imread(panels[lab]))
        ax.text(-0.01, 1.02, lab, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="right")
    fig.savefig(f"{FIGDIR}/Figure4_full.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{FIGDIR}/Figure4_full.pdf", bbox_inches="tight")
    plt.close(fig)


def digest():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        def hdr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)
        hdr("FIGURE 4 - RESULTS DIGEST (malignant programs)")
        try:
            import scanpy as sc
            n = sc.read_h5ad(f"{FIGDIR}/malignant_tumour_cells.h5ad", backed="r").n_obs
            print(f"Malignant cells: {n:,}")
        except Exception as e:
            print("malignant cell count unavailable:", e)
        rp = f"{FIGDIR}/programs/recurrent_programs_summary.csv"
        if os.path.exists(rp):
            df = pd.read_csv(rp)
            hdr("Recurrent programs")
            print(df.to_string(index=False))
        sc_by = f"{FIGDIR}/state_scores/state_scores_by_cancer.csv"
        if os.path.exists(sc_by):
            print("\nstate_scores_by_cancer.csv shape:", pd.read_csv(sc_by, index_col=0).shape)
    open(f"{FIGDIR}/Figure4_results.txt", "w").write(buf.getvalue())


def main():
    composite()
    digest()
    print("Figure 4 regenerated.")


if __name__ == "__main__":
    main()
