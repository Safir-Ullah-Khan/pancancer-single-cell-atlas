"""
Final manuscript figure: Supplementary Figure 4 (see FIGURE_MAP.md).
figure7.py - Clinical association of malignant programs (Fig 7).

Regenerates the 6-panel composite (a schematic, b HR heatmap, c KM curves,
d program summary, e forest plot [proliferation], f forest plot [hypoxia])
and the results digest from saved survival outputs.

UPSTREAM ANALYSIS (documented; not re-run here - see Methods):
  Malignant program scores related to survival in 17 TCGA cohorts
  (UCSC Xena; ~7,683 patients). Cox proportional-hazards per program/cohort
  (lifelines), HR per SD of program score, Benjamini-Hochberg correction;
  Kaplan-Meier for strongest associations. Univariate; bulk-vs-single-cell
  caveats noted in Methods. Products under tcga/ and survival/.

INPUTS (saved outputs):
  /data/paper_tumoronly/Figure7/panels/Fig7[a-f]_*.png
  /data/paper_tumoronly/Figure7/survival/cox_results.csv
  /data/paper_tumoronly/Figure7/survival/hr_matrix.csv
  /data/paper_tumoronly/Figure7/survival/sig_matrix.csv
OUTPUTS:
  /data/paper_tumoronly/Figure7/Figure7_full.{png,pdf}
  /data/paper_tumoronly/Figure7/Figure7_results.txt
"""
import os, sys, io, contextlib
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
sys.path.insert(0, os.path.dirname(__file__))
from atlas_setup import mm

FIGDIR = "/data/paper_tumoronly/Figure7"
PAN    = f"{FIGDIR}/panels"


def composite():
    panels = {"a": f"{PAN}/Fig7a_schematic.png", "b": f"{PAN}/Fig7b_hr_heatmap.png",
              "c": f"{PAN}/Fig7c_km.png", "d": f"{PAN}/Fig7d_summary.png",
              "e": f"{PAN}/Fig7e_forest_prolif.png", "f": f"{PAN}/Fig7f_forest_hypoxia.png"}
    missing = [k for k, v in panels.items() if not os.path.exists(v)]
    if missing: print("WARNING missing panels:", missing)

    fig = plt.figure(figsize=mm(183, 250))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.7, 1.0, 1.0, 1.0], hspace=0.12, wspace=0.06)
    layout = {"a": gs[0, :], "b": gs[1, :], "c": gs[2, 0],
              "d": gs[2, 1], "e": gs[3, 0], "f": gs[3, 1]}
    for lab, cell in layout.items():
        ax = fig.add_subplot(cell); ax.axis("off")
        if os.path.exists(panels[lab]): ax.imshow(mpimg.imread(panels[lab]))
        ax.text(-0.01, 1.02, lab, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="right")
    fig.savefig(f"{FIGDIR}/Figure7_full.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{FIGDIR}/Figure7_full.pdf", bbox_inches="tight")
    plt.close(fig)


def digest():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        def hdr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)
        hdr("FIGURE 7 - RESULTS DIGEST (clinical survival)")
        cox = f"{FIGDIR}/survival/cox_results.csv"
        if os.path.exists(cox):
            df = pd.read_csv(cox)
            print("cox_results.csv shape:", df.shape)
            qcol = [c for c in df.columns if c.lower() in ("q", "fdr", "padj", "q_value", "qvalue")]
            hrcol = [c for c in df.columns if "hr" in c.lower() or "hazard" in c.lower()]
            if qcol:
                sig = df[df[qcol[0]] < 0.05]
                print(f"\nSignificant associations (q<0.05): {len(sig)} of {len(df)}")
                if hrcol and len(sig):
                    s = sig.sort_values(hrcol[0], ascending=False)
                    print("\nStrongest adverse (top 5 by HR):")
                    print(s.head(5).to_string(index=False))
                    print("\nStrongest protective (bottom 5 by HR):")
                    print(s.tail(5).to_string(index=False))
    open(f"{FIGDIR}/Figure7_results.txt", "w").write(buf.getvalue())


def main():
    composite()
    digest()
    print("Figure 7 regenerated.")


if __name__ == "__main__":
    main()
