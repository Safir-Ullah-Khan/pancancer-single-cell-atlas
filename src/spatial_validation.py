#!/usr/bin/env python3
"""
spatial_validation.py - Spatial validation of discriminative gene panels.
Final manuscript figure: Supplementary Figure 6 (see FIGURE_MAP.md).

Tests whether the 25-gene discriminative panels derived from the single-cell
classifier resolve malignant identity in intact tissue. The panels are applied
without modification to public spatial transcriptomic sections, using the same
scoring procedure as the TCGA validation (tcga_validation.py): cohort-wide
gene-level z-scoring, panel score = mean z across panel genes, prediction =
highest-scoring panel.

Data are retrieved from the HEST-1k compendium (v1.3.0), a gated Hugging Face
dataset. Access must be requested at https://huggingface.co/datasets/MahmoodLab/hest
and authenticated before running (huggingface_hub.login).

Pipeline:
  1. query()     - select human Visium / Visium HD sections matching atlas cancers
  2. download()  - fetch the selected sections (st/*.h5ad only)
  3. panels()    - extract top-25 discriminative genes per cancer from classifier.pkl
  4. score()     - pool sections, z-score genes, score all panels at every spot
  5. figure()    - render the 6-panel composite

UPSTREAM DATA:
  Figure6/model/classifier.pkl   (trained LogisticRegression + feature names)

OUTPUTS (OUTDIR):
  spatial_sections.csv            - selected sections + provenance
  spatial_panels.csv              - the 25-gene panels used
  spatial_specificity.csv         - per-section: true cancer, prediction, matched rank
  spatial_specificity_matrix.csv  - 30 panels x 12 cancers mean z-score
  spatial_map_<id>.csv            - per-spot scores + coordinates (example sections)
  spatial_validation.png/pdf      - Supplementary Figure 6

Usage:
    python spatial_validation.py --step all
    python spatial_validation.py --step score      # single step
"""
import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---- paths ----
MODELDIR = "/data/paper_tumoronly/Figure6/model"
DATADIR = "/data/hest_data"
OUTDIR = "/data/spatial_validation"
FIGDIR = "/data/spatial_validation/figures"

HEST_REPO = "MahmoodLab/hest"
HEST_META = "HEST_v1_3_0.csv"

MIN_GENES = 17_000      # whole-transcriptome coverage of panel genes
MAX_PER_CANCER = 6      # prevent any one cancer dominating
MIN_COUNTS = 200        # per-spot count floor
MIN_SPOTS = 100         # per-section spot floor
MIN_PANEL_GENES = 10    # panel must retain this many genes after intersection
PANEL_SIZE = 25
EXAMPLE_SECTIONS = ["TENX29", "INT2", "NCBI643"]   # CRC, ccRCC, HCC

# OncoTree code -> atlas cancer type
ONCOTREE_MAP = {
    "COAD": "Colorectal Cancer", "READ": "Colorectal Cancer",
    "COADREAD": "Colorectal Cancer",
    "SCCRCC": "Clear Cell Renal Cell Carcinoma",
    "IDC": "Breast Cancer", "ILC": "Breast Cancer",
    "PRAD": "Prostate Cancer",
    "BLCA": "Bladder Cancer",
    "LUAD": "Lung Adenocarcinoma", "LUSC": "Lung Squamous Cell Carcinoma",
    "PAAD": "Pancreatic Ductal Adenocarcinoma",
    "PDAC": "Pancreatic Ductal Adenocarcinoma",
    "GBM": "Glioma",
    "HCC": "Hepatocellular Cancer",
    "CESC": "Cervical Cancer",
    "HGSOC": "Ovarian Carcinoma", "SOC": "Ovarian Carcinoma",
}

CANCER_ABBR = {
    "Colorectal Cancer": "CRC", "Clear Cell Renal Cell Carcinoma": "ccRCC",
    "Breast Cancer": "BRCA", "Prostate Cancer": "PRAD", "Bladder Cancer": "BLCA",
    "Lung Adenocarcinoma": "LUAD", "Lung Squamous Cell Carcinoma": "LUSC",
    "Pancreatic Ductal Adenocarcinoma": "PAAD", "Glioma": "GLIOMA",
    "Hepatocellular Cancer": "HCC", "Ovarian Carcinoma": "OV",
    "Cervical Cancer": "CESC", "Appendiceal Neoplasms": "APPN",
    "Basal Cell Carcinoma": "BCC", "Neuroblastoma": "NB",
    "Esophageal Adenocarcinoma": "EAC",
    "Esophageal Squamous Cell Carcinoma": "ESCC",
    "Head and Neck Squamous Cell Carcinoma": "HNSC",
    "Cutaneous Squamous Cell Carcinoma": "cSCC",
    "Oral Squamous Cell Carcinoma": "OSCC",
    "Non-small Cell Lung Cancer": "NSCLC",
    "Mixed-phenotype Acute Leukemia": "MPAL", "Retinoblastoma": "RB",
    "Salivary Adenoid Cystic Carcinoma": "SACC",
    "Squamous Cell Carcinoma in Situ": "SCCIS",
    "Gallbladder Carcinoma": "GBC", "Papillary Thyroid Carcinoma": "THCA",
    "Papillary Renal Cell Carcinoma": "pRCC",
    "Distal Cholangiocarcinoma": "dCCA", "Uterine Leiomyoma": "ULM",
}
abbr = lambda c: CANCER_ABBR.get(c, c[:8])


# =====================================================================
# 1. section selection
# =====================================================================
def query():
    """Select human Visium/Visium HD sections matching atlas cancer types."""
    meta = pd.read_csv(f"hf://datasets/{HEST_REPO}/{HEST_META}")
    v = meta[
        (meta.species == "Homo sapiens")
        & (meta.st_technology.str.contains("Visium", case=False, na=False))
        & (meta.oncotree_code.isin(ONCOTREE_MAP))
        & (meta.nb_genes >= MIN_GENES)
    ].copy()
    v["atlas_cancer"] = v["oncotree_code"].map(ONCOTREE_MAP)

    sel = (v.sort_values(["nb_genes", "spots_under_tissue"], ascending=False)
             .groupby("atlas_cancer").head(MAX_PER_CANCER))
    cols = ["id", "atlas_cancer", "oncotree_code", "st_technology", "nb_genes",
            "spots_under_tissue", "patient", "dataset_title", "study_link", "license"]
    sel = sel[[c for c in cols if c in sel.columns]]

    os.makedirs(OUTDIR, exist_ok=True)
    sel.to_csv(f"{OUTDIR}/spatial_sections.csv", index=False)
    print(f"selected {len(sel)} sections across {sel.atlas_cancer.nunique()} cancers")
    print(sel.groupby(["atlas_cancer", "st_technology"]).size().to_string())
    return sel


def download():
    """Fetch st/ h5ad objects for the selected sections (whole-slide images skipped)."""
    from huggingface_hub import snapshot_download

    sel = pd.read_csv(f"{OUTDIR}/spatial_sections.csv")
    ids = sel["id"].tolist()
    print(f"downloading {len(ids)} sections to {DATADIR} ...")
    snapshot_download(repo_id=HEST_REPO, repo_type="dataset", local_dir=DATADIR,
                      allow_patterns=[f"st/{i}.h5ad" for i in ids])
    import glob
    fs = glob.glob(f"{DATADIR}/st/*.h5ad")
    print(f"done: {len(fs)} files, {sum(os.path.getsize(f) for f in fs)/1e9:.1f} GB")


# =====================================================================
# 2. panels
# =====================================================================
def panels(n=PANEL_SIZE):
    """Top-n discriminative genes per cancer, ranked by classifier coefficient."""
    import joblib
    obj = joblib.load(f"{MODELDIR}/classifier.pkl")
    clf, feats = obj["clf"], np.array([str(g) for g in obj["features"]])
    p = {c: list(feats[np.argsort(clf.coef_[i])[::-1]][:n])
         for i, c in enumerate(clf.classes_)}
    os.makedirs(OUTDIR, exist_ok=True)
    pd.DataFrame([{"cancer": c, "rank": i + 1, "gene": g}
                  for c, gs in p.items() for i, g in enumerate(gs)]
                 ).to_csv(f"{OUTDIR}/spatial_panels.csv", index=False)
    print(f"extracted {len(p)} panels x {n} genes")
    return p


# =====================================================================
# 3. scoring
# =====================================================================
def _load_sections(sections):
    """Load, filter and log-normalize each section; return list of AnnData."""
    import scanpy as sc
    mats, meta, coords = [], [], []
    for s in sections.itertuples():
        p = f"{DATADIR}/st/{s.id}.h5ad"
        if not os.path.exists(p):
            print(f"  missing: {s.id}")
            continue
        ad = sc.read_h5ad(p)
        ad.var_names_make_unique()
        if "in_tissue" in ad.obs:
            ad = ad[ad.obs["in_tissue"] == 1].copy()
        sc.pp.filter_cells(ad, min_counts=MIN_COUNTS)
        if ad.n_obs < MIN_SPOTS:
            print(f"  skip {s.id}: {ad.n_obs} spots")
            continue
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
        mats.append(ad)
        meta += [(s.id, s.atlas_cancer)] * ad.n_obs
        coords.append(ad.obsm["spatial"])
        print(f"  {s.id}: {ad.n_obs} spots")
    return mats, pd.DataFrame(meta, columns=["id", "true_cancer"]), np.vstack(coords)


def score():
    """Pool sections, z-score genes cohort-wide, score all panels at every spot."""
    sections = pd.read_csv(f"{OUTDIR}/spatial_sections.csv")
    P = {c: [str(g) for g in d["gene"]]
         for c, d in pd.read_csv(f"{OUTDIR}/spatial_panels.csv").groupby("cancer")}

    mats, sm, xy = _load_sections(sections)
    sm["x"], sm["y"] = xy[:, 0], xy[:, 1]

    common = sorted(set.intersection(*[set(a.var_names) for a in mats]))
    X = np.vstack([np.asarray(a[:, common].X.todense()
                              if hasattr(a[:, common].X, "todense") else a[:, common].X)
                   for a in mats])
    pooled = pd.DataFrame(X.T, index=common)
    print(f"\npooled: {pooled.shape[1]:,} spots x {len(common):,} common genes")

    # cohort-wide gene-level z-score (as in tcga_validation.py)
    zg = pooled.sub(pooled.mean(axis=1), axis=0).div(
        pooled.std(axis=1).replace(0, np.nan), axis=0)

    testable = [c for c, g in P.items()
                if len([x for x in g if x in zg.index]) >= MIN_PANEL_GENES]
    sc_df = pd.DataFrame(index=sm.index, columns=testable, dtype=float)
    for c in testable:
        sc_df[c] = zg.loc[[g for g in P[c] if g in zg.index]].mean(axis=0).values
    print(f"scored {len(testable)} panels")

    # per-spot
    sm["pred"] = sc_df.idxmax(axis=1)
    ev = sm[sm.true_cancer.isin(testable)]
    acc_spot = (ev.pred == ev.true_cancer).mean() * 100

    # per-section
    sec = sc_df.groupby(sm["id"]).mean()
    true = sm.groupby("id")["true_cancer"].first()
    pred = sec.idxmax(axis=1)
    rank = pd.Series({i: int(sec.loc[i].rank(ascending=False)[true[i]])
                      for i in sec.index if true[i] in testable})
    acc_sec = (pred[rank.index] == true[rank.index]).mean() * 100
    rand = 100 / len(testable)

    print("\n" + "=" * 62)
    print(f"per-spot    {acc_spot:5.1f}%  ({len(ev):,} spots, {acc_spot/rand:.1f}x random)")
    print(f"per-section {acc_sec:5.1f}%  ({len(rank)} sections, {acc_sec/rand:.1f}x), "
          f"median matched rank {rank.median():.0f}")
    print("=" * 62)

    out = pd.DataFrame({"id": rank.index, "true_cancer": true[rank.index].values,
                        "pred": pred[rank.index].values, "matched_rank": rank.values})
    out["correct"] = out.pred == out.true_cancer
    out.to_csv(f"{OUTDIR}/spatial_specificity.csv", index=False)
    print(out.groupby("true_cancer").agg(n=("id", "count"), correct=("correct", "sum"),
          median_rank=("matched_rank", "median")).to_string())

    # panels x true-cancer matrix (Supp Fig 6a)
    sc_df.groupby(sm["true_cancer"]).mean().T.to_csv(
        f"{OUTDIR}/spatial_specificity_matrix.csv")

    # per-spot maps for the example sections (Supp Fig 6d-f)
    for sid in EXAMPLE_SECTIONS:
        m = sm.id == sid
        if not m.any():
            continue
        tc = sm.loc[m, "true_cancer"].iloc[0]
        d = sm.loc[m, ["x", "y"]].copy()
        d["matched_score"] = sc_df.loc[m, tc].values
        d["pred"] = sc_df.loc[m].idxmax(axis=1).values
        d["true_cancer"] = tc
        if "EPCAM" in zg.index:
            d["EPCAM"] = zg.loc["EPCAM"].values[m.values]
        d.to_csv(f"{OUTDIR}/spatial_map_{sid}.csv", index=False)
    print(f"\nsaved results to {OUTDIR}/")


# =====================================================================
# 4. figure
# =====================================================================
def figure():
    """Supplementary Figure 6: specificity, per-cancer accuracy, rank, spatial maps."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy.stats import spearmanr

    mpl.rcParams.update({
        "font.size": 12, "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.3, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.labelsize": 13, "pdf.fonttype": 42,
    })
    res = pd.read_csv(f"{OUTDIR}/spatial_specificity.csv")
    spec = pd.read_csv(f"{OUTDIR}/spatial_specificity_matrix.csv", index_col=0)

    fig = plt.figure(figsize=(18, 11.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.40, left=0.07,
                  right=0.97, top=0.89, bottom=0.09, height_ratios=[1.35, 1.0])
    plab = lambda ax, l: ax.text(-0.20, 1.07, l, transform=ax.transAxes,
                                 fontsize=20, fontweight="bold", va="bottom", ha="right")

    # a) specificity heatmap
    ax = fig.add_subplot(gs[0, 0]); plab(ax, "a")
    cols = list(spec.columns)
    rows = [c for c in cols if c in spec.index] + [c for c in spec.index if c not in cols]
    M = spec.loc[rows, cols]
    vm = np.nanmax(np.abs(M.values))
    im = ax.imshow(M.values, cmap="RdBu_r", aspect="auto", vmin=-vm, vmax=vm)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([abbr(c) for c in cols], rotation=90, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([abbr(c) for c in rows], fontsize=8.5)
    for j, c in enumerate(cols):
        if c in rows:
            ax.add_patch(plt.Rectangle((j - 0.5, rows.index(c) - 0.5), 1, 1,
                                       fill=False, ec="black", lw=2.0))
    ax.set_xlabel("Spatial section (true cancer)", fontweight="bold", fontsize=12)
    ax.set_ylabel("scRNA-derived panel", fontweight="bold", fontsize=12)
    ax.set_title("Panels score highest in matched cancer", fontsize=13,
                 fontweight="bold", loc="left", pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("mean panel z-score", fontsize=10, fontweight="bold")

    # b) per-cancer accuracy
    ax = fig.add_subplot(gs[0, 1]); plab(ax, "b")
    pc = res.groupby("true_cancer").agg(n=("id", "count"), correct=("correct", "sum")).reset_index()
    pc["pct"] = 100 * pc.correct / pc.n
    pc = pc.sort_values("pct")
    y = np.arange(len(pc))
    col = ["#238b45" if v >= 80 else ("#f0a202" if v >= 40 else "#c0392b") for v in pc.pct]
    ax.barh(y, pc.pct, color=col, height=0.72, edgecolor="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{abbr(c)} (n={int(n)})" for c, n in zip(pc.true_cancer, pc.n)],
                       fontsize=11, fontweight="bold")
    ax.axvline(100 / len(spec.index), ls=":", c="#666", lw=1.6)
    ax.set_xlabel("Sections correctly classified (%)", fontweight="bold", fontsize=12)
    ax.set_xlim(0, 105)
    ax.set_title("Per-cancer self-specificity", fontsize=13, fontweight="bold",
                 loc="left", pad=12)

    # c) matched rank
    ax = fig.add_subplot(gs[0, 2]); plab(ax, "c")
    order = res.groupby("true_cancer")["matched_rank"].median().sort_values().index
    rng = np.random.default_rng(0)
    for i, c in enumerate(order):
        v = res[res.true_cancer == c]["matched_rank"].values
        ax.scatter(rng.normal(i, 0.10, len(v)), v, s=55, c="#3B6FB6", alpha=0.8,
                   edgecolors="white", lw=0.8, zorder=3)
        ax.hlines(np.median(v), i - 0.30, i + 0.30, colors="black", lw=2.8, zorder=4)
    ax.axhline(1, ls="--", c="#238b45", lw=1.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([abbr(c) for c in order], rotation=90, fontsize=11, fontweight="bold")
    ax.set_ylabel("Rank of matched panel (of 30)", fontweight="bold", fontsize=12)
    ax.set_yscale("log"); ax.set_yticks([1, 2, 5, 10, 20, 30])
    ax.set_yticklabels([1, 2, 5, 10, 20, 30], fontsize=11)
    ax.set_title(f"Median matched rank = {int(res.matched_rank.median())}",
                 fontsize=13, fontweight="bold", loc="left", pad=12)

    # d-f) spatial maps
    for k, sid in enumerate(EXAMPLE_SECTIONS):
        ax = fig.add_subplot(gs[1, k]); plab(ax, "def"[k])
        p = f"{OUTDIR}/spatial_map_{sid}.csv"
        if not os.path.exists(p):
            ax.axis("off"); continue
        d = pd.read_csv(p)
        tc = d["true_cancer"].iloc[0]
        s_ = ax.scatter(d.x, -d.y, c=d.matched_score, s=9, cmap="magma",
                        vmin=np.percentile(d.matched_score, 2),
                        vmax=np.percentile(d.matched_score, 98), edgecolors="none")
        ax.set_aspect("equal"); ax.axis("off")
        sub = f"{abbr(tc)}  ({sid})"
        if "EPCAM" in d.columns:
            r, _ = spearmanr(d.matched_score, d.EPCAM)
            sub += f"\nvs EPCAM:  rho = {r:.2f}"
        ax.set_title(sub, fontsize=12, fontweight="bold", loc="left", pad=10)
        cb = fig.colorbar(s_, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label("matched panel score", fontsize=10, fontweight="bold")

    n_ok, n_tot = int(res.correct.sum()), len(res)
    fig.suptitle("Single-cell-derived gene panels resolve tumour identity in intact tissue\n"
                 f"{n_ok}/{n_tot} sections correctly classified ({100*n_ok/n_tot:.1f}%)",
                 fontsize=16, fontweight="bold", y=0.975, linespacing=1.5)

    os.makedirs(FIGDIR, exist_ok=True)
    out = f"{FIGDIR}/spatial_validation"
    fig.savefig(out + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}.png / .pdf")


# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step", default="all",
                    choices=["all", "query", "download", "panels", "score", "figure"])
    a = ap.parse_args()
    if a.step in ("all", "query"):
        query()
    if a.step in ("all", "download"):
        download()
    if a.step in ("all", "panels"):
        panels()
    if a.step in ("all", "score"):
        score()
    if a.step in ("all", "figure"):
        figure()


if __name__ == "__main__":
    main()
