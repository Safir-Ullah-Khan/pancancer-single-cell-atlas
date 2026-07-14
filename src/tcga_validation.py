"""
tcga_validation.py - Cross-platform validation of single-cell cancer signatures.

Tests whether malignant discriminative-gene panels learned from the single-cell
atlas classify INDEPENDENT bulk TCGA patients to their cancer of origin.
Panel-size sweep -> 25-gene panels -> specificity across cohorts ->
per-patient confusion (74.9% acc, 12.7x random) -> per-cancer recall.

UPSTREAM DATA (see Methods):
  Figure6/model/classifier.pkl   (trained LogisticRegression + feature names)
  Figure7/tcga/*_expr.gz         (17 TCGA cohorts, log2 RSEM, UCSC Xena)
  Figure7/tcga/cohort_map.json
OUTPUTS (DATADIR): specificity_matrix.csv, panel_specificity_summary.csv,
  tcga_confusion_{counts,pct}.csv, panel_size_sweep.csv, tcga_validation_full.{png,pdf}
Final manuscript figure: Figure 6 (see FIGURE_MAP.md)
"""
import os, sys, json, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
sys.path.insert(0, os.path.dirname(__file__))
from atlas_setup import mm

DATADIR = "/data/paper_tumoronly/Figure8_validation"
MODELDIR = "/data/paper_tumoronly/Figure6/model"
TCGA = "/data/paper_tumoronly/Figure7/tcga"
N_PRIMARY = 25


def _load_tcga_pooled():
    import joblib
    obj = joblib.load(f"{MODELDIR}/classifier.pkl")
    clf, feats = obj["clf"], np.array(obj["features"])
    ranked = {c: feats[np.argsort(clf.coef_[i])[::-1]].tolist()
              for i, c in enumerate(clf.classes_)}
    cm = json.load(open(f"{TCGA}/cohort_map.json"))
    sc2tcga = {}
    for k, v in cm.items():
        sc2tcga.setdefault(v, k)
    sc2tcga.update({"Esophageal Squamous Cell Carcinoma": "ESCA",
                    "Esophageal Adenocarcinoma": "ESCA",
                    "Gallbladder Carcinoma": "CHOL",
                    "Distal Cholangiocarcinoma": "CHOL"})
    have = {os.path.basename(f).split("_expr")[0] for f in glob.glob(f"{TCGA}/*_expr.gz")}

    def load_expr(code):
        df = pd.read_csv(f"{TCGA}/{code}_expr.gz", sep="\t", index_col=0)
        tum = [c for c in df.columns
               if c.split("-")[3][:2] in ("01","02","03","04","05","06","07","08","09")]
        return df[tum] if tum else df

    mats, labels = [], []
    for code in sorted(have):
        e = load_expr(code); e.columns = [f"{code}|{c}" for c in e.columns]
        mats.append(e); labels += [code] * e.shape[1]
    common = sorted(set(mats[0].index).intersection(*[set(m.index) for m in mats[1:]]))
    pooled = pd.concat([m.loc[common] for m in mats], axis=1)
    pc = pd.Series(labels, index=pooled.columns)
    zg = pooled.sub(pooled.mean(axis=1), axis=0).div(pooled.std(axis=1).replace(0, np.nan), axis=0)
    return ranked, sc2tcga, zg, pc, sorted(set(labels))


def run_analysis(N=N_PRIMARY):
    """Recompute specificity + confusion at panel size N; write CSVs. (slow)"""
    os.makedirs(DATADIR, exist_ok=True)
    ranked, sc2tcga, zg, pc, labels = _load_tcga_pooled()
    testable, pan = [], {}
    for c in ranked:
        if c not in sc2tcga or sc2tcga[c] not in set(labels):
            continue
        genes = [g for g in ranked[c] if g in zg.index][:N]
        if len(genes) >= 5:
            pan[c] = genes; testable.append(c)
    score = pd.DataFrame(index=zg.columns, columns=testable, dtype=float)
    for c in testable:
        score[c] = zg.loc[pan[c]].mean(axis=0)
    codes = sorted(set(labels))
    spec = pd.DataFrame(index=testable, columns=codes, dtype=float)
    for c in testable:
        spec.loc[c] = score[c].groupby(pc).mean()
    rows = []
    for c in testable:
        own = sc2tcga[c]; row = spec.loc[c].dropna()
        rows.append({"scRNA_panel": c, "matched_TCGA": own, "coverage": f"{len(pan[c])}/{N}",
                     "matched_rank": int(row.rank(ascending=False)[own]),
                     "top_cohort": row.idxmax(), "self_specific": row.idxmax() == own})
    spec.to_csv(f"{DATADIR}/specificity_matrix.csv")
    pd.DataFrame(rows).to_csv(f"{DATADIR}/panel_specificity_summary.csv", index=False)
    pred = score.idxmax(axis=1).map(sc2tcga)
    ccodes = sorted(set(sc2tcga[c] for c in testable))
    conf = pd.crosstab(pc, pred).reindex(index=ccodes, columns=ccodes, fill_value=0)
    conf.to_csv(f"{DATADIR}/tcga_confusion_counts.csv")
    (conf.div(conf.sum(axis=1), axis=0) * 100).to_csv(f"{DATADIR}/tcga_confusion_pct.csv")
    acc = 100 * np.trace(conf.values) / conf.values.sum()
    print(f"N={N}: accuracy {acc:.1f}% ({acc/(100/len(ccodes)):.1f}x), "
          f"self-specific {sum(r['self_specific'] for r in rows)}/{len(rows)}")


def composite():
    """Tile saved panels into the validation composite."""
    panels = {"a": "Fig8a_schematic.png", "b": "panel_specificity_heatmap.png",
              "c": "panel_transfer_rank.png", "d": "panel_size_sweep_plot.png",
              "e": "panel_confusion.png", "f": "panel_recall.png"}
    fig = plt.figure(figsize=mm(183, 245))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.5, 1.3, 1.3],
                          width_ratios=[1.15, 0.95, 0.95], hspace=0.16, wspace=0.12)
    layout = {"a": gs[0, :], "b": gs[1, 0], "c": gs[1, 1], "d": gs[1, 2],
              "e": gs[2, 0:2], "f": gs[2, 2]}
    for lab, cell in layout.items():
        ax = fig.add_subplot(cell); ax.axis("off")
        p = f"{DATADIR}/{panels[lab]}"
        if os.path.exists(p):
            ax.imshow(mpimg.imread(p))
        ax.text(-0.01, 1.02, lab, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="right")
    fig.savefig(f"{DATADIR}/tcga_validation_full.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{DATADIR}/tcga_validation_full.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    # run_analysis(N_PRIMARY)   # uncomment to recompute from TCGA (slow)
    composite()
    print("tcga_validation figure regenerated.")


if __name__ == "__main__":
    main()
