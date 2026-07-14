
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from atlas_setup import (
    CANCER_ORDER, CANCER_ABBR, CANCER_COLORS,
    set_nature_style, mm, save_fig, SINGLE_COL_MM, DOUBLE_COL_MM,
)

FULL_H5AD   = "/data/final/cancerscem_final.h5ad"
PERSAMPLE   = "/data/h5ad_per_sample"
OUTDIR      = "/data/figures"
COUNTS_CSV  = OUTDIR + "/fig1_counts_by_cancer.csv"
QC_CSV      = OUTDIR + "/fig1_qc_percell.csv"
MT_PREFIX   = "MT-"
SAMPLE_TYPE_COLORS = {"Tumour": "#B2182B", "Normal": "#2166AC"}
QC_OBS_COLS = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]

def load_full_obs():
    import scanpy as sc
    print("reading obs (backed) from " + FULL_H5AD + " ...")
    ad_full = sc.read_h5ad(FULL_H5AD, backed="r")
    obs = ad_full.obs.copy()
    print("obs columns:", list(obs.columns))
    print("n cells:", obs.shape[0])
    return obs, ad_full

def build_counts_table(obs):
    ct = pd.crosstab(obs["cancer_type"], obs["sample_type"])
    for col in ["Tumour", "Normal"]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[["Tumour", "Normal"]]
    ct["total"] = ct.sum(axis=1)
    ct["pct_tumour"] = 100 * ct["Tumour"] / ct["total"]
    order = [c for c in CANCER_ORDER if c in ct.index]
    missing = [c for c in ct.index if c not in CANCER_ORDER]
    if missing:
        print("WARNING: cancer names not in CANCER_ORDER:", missing)
        order = order + missing
    ct = ct.loc[order]
    ct.to_csv(COUNTS_CSV)
    print("saved " + COUNTS_CSV)
    return ct

def get_qc(obs, allow_persample=True):
    if os.path.exists(QC_CSV):
        print("loading cached QC from " + QC_CSV)
        return pd.read_csv(QC_CSV)
    have = [c for c in QC_OBS_COLS if c in obs.columns]
    if len(have) == 3:
        print("QC columns already in full obs.")
        qc = obs[["cancer_type", "sample_type"] + QC_OBS_COLS].copy()
        qc.columns = ["cancer_type", "sample_type", "n_genes", "n_umi", "pct_mt"]
    elif allow_persample:
        print("QC not in obs (found " + str(have) + "); computing from per-sample files.")
        qc = _qc_from_persample()
    else:
        raise RuntimeError("QC not in obs and per-sample pass disabled.")
    qc.to_csv(QC_CSV, index=False)
    print("saved " + QC_CSV)
    return qc

def _qc_from_persample():
    import glob, scanpy as sc
    rows = []
    files = sorted(glob.glob(PERSAMPLE + "/*.h5ad"))
    print("computing QC across " + str(len(files)) + " per-sample files ...")
    for i, f in enumerate(files):
        a = sc.read_h5ad(f)
        a.var["mt"] = a.var_names.str.startswith(MT_PREFIX)
        sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True, percent_top=None)
        sub = pd.DataFrame({
            "cancer_type": a.obs.get("cancer_type", pd.Series(["NA"] * a.n_obs)).values,
            "sample_type": a.obs.get("sample_type", pd.Series(["NA"] * a.n_obs)).values,
            "n_genes": a.obs["n_genes_by_counts"].values,
            "n_umi":   a.obs["total_counts"].values,
            "pct_mt":  a.obs["pct_counts_mt"].values,
        })
        rows.append(sub)
        if (i + 1) % 100 == 0:
            print("  " + str(i+1) + "/" + str(len(files)))
    return pd.concat(rows, ignore_index=True)

def _abbr(name):
    return CANCER_ABBR.get(name, name)

def plot_1c(ct, ax):
    order = ct.sort_values("total").index
    y = np.arange(len(order))
    vals = ct.loc[order, "total"].values
    colors = [CANCER_COLORS.get(c, "#888888") for c in order]
    ax.barh(y, vals, color=colors, edgecolor="none", height=0.78)
    ax.set_yticks(y); ax.set_yticklabels([_abbr(c) for c in order])
    ax.set_xscale("log"); ax.set_xlabel("Cells (n)")
    ax.set_xlim(left=max(1, vals.min() * 0.6))
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("Cells per cancer type", loc="left", fontweight="bold")
    return order

def plot_1d(ct, ax, order=None):
    if order is None:
        order = ct.sort_values("total").index
    y = np.arange(len(order))
    tum = ct.loc[order, "pct_tumour"].values
    nor = 100 - tum
    ax.barh(y, tum, color=SAMPLE_TYPE_COLORS["Tumour"], height=0.78, label="Tumour")
    ax.barh(y, nor, left=tum, color=SAMPLE_TYPE_COLORS["Normal"], height=0.78, label="Normal")
    ax.set_yticks(y); ax.set_yticklabels([_abbr(c) for c in order])
    ax.set_xlim(0, 100); ax.set_xlabel("Composition (%)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Tumour / normal composition", loc="left", fontweight="bold")

def plot_1e(qc, axes, plot_subsample=4000):
    metrics = [("n_genes", "Genes / cell", True),
               ("n_umi",   "UMIs / cell",  True),
               ("pct_mt",  "% mito",       False)]
    order = [c for c in CANCER_ORDER if c in qc["cancer_type"].unique()]
    rng = np.random.default_rng(0)
    for ax, (col, label, logy) in zip(axes, metrics):
        data, colors = [], []
        for c in order:
            v = qc.loc[qc["cancer_type"] == c, col].dropna().values
            if len(v) > plot_subsample:
                v = rng.choice(v, plot_subsample, replace=False)
            data.append(v); colors.append(CANCER_COLORS.get(c, "#888888"))
        bp = ax.boxplot(data, positions=np.arange(len(order)), widths=0.6,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color="black", linewidth=0.6),
                        whiskerprops=dict(linewidth=0.4),
                        capprops=dict(linewidth=0.4),
                        boxprops=dict(linewidth=0.4))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_edgecolor("black")
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([_abbr(c) for c in order], rotation=90)
        ax.tick_params(axis="x", length=0)
    axes[0].set_title("Per-cell QC by cancer type", loc="left", fontweight="bold")

def render_cd(ct):
    set_nature_style()
    fig, (axc, axd) = plt.subplots(1, 2, figsize=mm(DOUBLE_COL_MM, 95), gridspec_kw=dict(wspace=0.55))
    order = plot_1c(ct, axc); plot_1d(ct, axd, order=order)
    save_fig(fig, OUTDIR + "/figure1_CD")
    return fig

def render_e(qc):
    set_nature_style()
    fig, axes = plt.subplots(3, 1, figsize=mm(DOUBLE_COL_MM, 130), sharex=True, gridspec_kw=dict(hspace=0.15))
    plot_1e(qc, axes)
    save_fig(fig, OUTDIR + "/figure1_E")
    return fig
