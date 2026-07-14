"""
Final manuscript figure: Figure 5 (see FIGURE_MAP.md).
figure6.py - Machine-learning classification of tumour identity (Fig 6).

Regenerates the 6-panel composite (a schematic, b confusion matrix, c
discriminative genes, d per-cancer performance, e feature-set comparison,
f cancer taxonomy) and the results digest from saved model outputs.

UPSTREAM ANALYSIS (documented; not re-run here - see Methods):
  Multinomial logistic-regression classifier on 89,530 malignant cells
  (2,000 HVG, standardized, stratified 70/30 split). Balanced accuracy /
  macro-F1, per-cancer metrics, confusion matrix, feature-set comparison,
  and a pseudobulk-similarity cancer taxonomy. Within-cohort evaluation only.
  Products under model/.

INPUTS (saved outputs):
  /data/paper_tumoronly/Figure6/panels/Fig6[a-f]_*.png
  /data/paper_tumoronly/Figure6/model/per_cancer_performance.csv
  /data/paper_tumoronly/Figure6/model/discriminative_genes.csv
  /data/paper_tumoronly/Figure6/model/featureset_comparison.csv
OUTPUTS:
  /data/paper_tumoronly/Figure6/Figure6_full.{png,pdf}
  /data/paper_tumoronly/Figure6/Figure6_results.txt
"""
import os, sys, io, contextlib
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
sys.path.insert(0, os.path.dirname(__file__))
from atlas_setup import mm

FIGDIR = "/data/paper_tumoronly/Figure6"
PAN    = f"{FIGDIR}/panels"


def composite():
    panels = {"a": f"{PAN}/Fig6a_schematic.png", "b": f"{PAN}/Fig6b_confusion.png",
              "c": f"{PAN}/Fig6c_genes.png", "d": f"{PAN}/Fig6d_percancer.png",
              "e": f"{PAN}/Fig6e_featuresets.png", "f": f"{PAN}/Fig6f_taxonomy.png"}
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
    fig.savefig(f"{FIGDIR}/Figure6_full.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{FIGDIR}/Figure6_full.pdf", bbox_inches="tight")
    plt.close(fig)


def digest():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        def hdr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)
        hdr("FIGURE 6 - RESULTS DIGEST (ML classifier)")
        perf = f"{FIGDIR}/model/per_cancer_performance.csv"
        if os.path.exists(perf):
            df = pd.read_csv(perf)
            print("per_cancer_performance.csv shape:", df.shape)
            fcol = [c for c in df.columns if "f1" in c.lower()]
            if fcol:
                s = df.sort_values(fcol[0], ascending=False)
                print("\nBest-classified (top 5):")
                print(s.head(5).to_string(index=False))
                print("\nHardest (bottom 5):")
                print(s.tail(5).to_string(index=False))
        fs = f"{FIGDIR}/model/featureset_comparison.csv"
        if os.path.exists(fs):
            hdr("Feature-set comparison")
            print(pd.read_csv(fs).to_string(index=False))
    open(f"{FIGDIR}/Figure6_results.txt", "w").write(buf.getvalue())


def main():
    composite()
    digest()
    print("Figure 6 regenerated.")


if __name__ == "__main__":
    main()
