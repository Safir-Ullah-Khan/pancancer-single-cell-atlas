
"""atlas_setup.py — shared figure + annotation utilities for the pan-cancer atlas."""
import numpy as np

CANCER_ORDER = [
    "Pancreatic Ductal Adenocarcinoma", "Esophageal Squamous Cell Carcinoma",
    "Prostate Cancer", "Clear Cell Renal Cell Carcinoma", "Lung Adenocarcinoma",
    "Neuroblastoma", "Colorectal Cancer", "Basal Cell Carcinoma", "Breast Cancer",
    "Glioma", "Lung Squamous Cell Carcinoma", "Hepatocellular Cancer",
    "Appendiceal Neoplasms", "Bladder Cancer", "Retinoblastoma", "Head and Neck SCC",
    "Esophageal Adenocarcinoma", "Uterine Leiomyoma", "Salivary Adenoid Cystic Carcinoma",
    "Cervical Cancer", "Ovarian Carcinoma", "Non-small Cell Lung Cancer", "Cutaneous SCC",
    "Mixed-phenotype Acute Leukemia", "Oral SCC", "Squamous Cell Carcinoma in Situ",
    "Gallbladder Carcinoma", "Papillary Thyroid Carcinoma", "Papillary Renal Cell Carcinoma",
    "Distal Cholangiocarcinoma",
]
CANCER_ABBR = {
    "Head and Neck Squamous Cell Carcinoma":"HNSC",
    "Cutaneous Squamous Cell Carcinoma":"cSCC",
    "Oral Squamous Cell Carcinoma":"OSCC",
    "Pancreatic Ductal Adenocarcinoma":"PAAD","Esophageal Squamous Cell Carcinoma":"ESCC",
    "Prostate Cancer":"PRAD","Clear Cell Renal Cell Carcinoma":"ccRCC","Lung Adenocarcinoma":"LUAD",
    "Neuroblastoma":"NB","Colorectal Cancer":"CRC","Basal Cell Carcinoma":"BCC","Breast Cancer":"BRCA",
    "Glioma":"GLIOMA","Lung Squamous Cell Carcinoma":"LUSC","Hepatocellular Cancer":"HCC",
    "Appendiceal Neoplasms":"APPN","Bladder Cancer":"BLCA","Retinoblastoma":"RB","Head and Neck SCC":"HNSC",
    "Esophageal Adenocarcinoma":"EAC","Uterine Leiomyoma":"ULM","Salivary Adenoid Cystic Carcinoma":"SACC",
    "Cervical Cancer":"CESC","Ovarian Carcinoma":"OV","Non-small Cell Lung Cancer":"NSCLC","Cutaneous SCC":"cSCC",
    "Mixed-phenotype Acute Leukemia":"MPAL","Oral SCC":"OSCC","Squamous Cell Carcinoma in Situ":"SCCIS",
    "Gallbladder Carcinoma":"GBC","Papillary Thyroid Carcinoma":"THCA","Papillary Renal Cell Carcinoma":"pRCC",
    "Distal Cholangiocarcinoma":"dCCA",
}
NONSTANDARD_ABBR = {
    "NB":"Neuroblastoma","BCC":"Basal Cell Carcinoma","APPN":"Appendiceal Neoplasms","RB":"Retinoblastoma",
    "EAC":"Esophageal Adenocarcinoma","ULM":"Uterine Leiomyoma","SACC":"Salivary Adenoid Cystic Carcinoma",
    "NSCLC":"Non-small Cell Lung Cancer (NOS)","cSCC":"Cutaneous Squamous Cell Carcinoma",
    "MPAL":"Mixed-phenotype Acute Leukemia","OSCC":"Oral Squamous Cell Carcinoma",
    "SCCIS":"Squamous Cell Carcinoma in Situ","GBC":"Gallbladder Carcinoma","dCCA":"Distal Cholangiocarcinoma",
    "ESCC":"Esophageal Squamous Cell Carcinoma","HCC":"Hepatocellular Carcinoma",
    "ccRCC":"Clear Cell Renal Cell Carcinoma","pRCC":"Papillary Renal Cell Carcinoma","CRC":"Colorectal Cancer",
}
_PALETTE30 = [
    "#E6194B","#3CB44B","#4363D8","#F58231","#911EB4","#42D4F4","#F032E6","#BFEF45","#469990","#9A6324",
    "#800000","#808000","#000075","#FF4500","#2F4F4F","#DAA520","#8B008B","#556B2F","#FF1493","#00CED1",
    "#B22222","#5F9EA0","#D2691E","#9ACD32","#6A5ACD","#C71585","#2E8B57","#FF8C00","#1E90FF","#A0522D",
]
assert len(_PALETTE30) == len(CANCER_ORDER) == 30
CANCER_COLORS = dict(zip(CANCER_ORDER, _PALETTE30))
SINGLE_COL_MM, DOUBLE_COL_MM = 89, 183

def set_nature_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
        "font.size":7,"axes.titlesize":7,"axes.labelsize":7,"xtick.labelsize":6,"ytick.labelsize":6,
        "legend.fontsize":6,"axes.linewidth":0.5,"xtick.major.width":0.5,"ytick.major.width":0.5,
        "xtick.major.size":2,"ytick.major.size":2,"axes.spines.top":False,"axes.spines.right":False,
        "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","figure.dpi":300,"savefig.dpi":300,
        "savefig.bbox":"tight","savefig.pad_inches":0.02,
    })

def mm(width_mm, height_mm=None):
    w = width_mm/25.4
    return w if height_mm is None else (w, height_mm/25.4)

def save_fig(fig, path_noext, dpi=300):
    fig.savefig(path_noext + ".png", dpi=dpi)
    fig.savefig(path_noext + ".pdf")
    print("saved " + path_noext + ".png  +  " + path_noext + ".pdf")

MARKER_SETS = {
    "T_cell":["CD3D","CD3E","CD3G","TRAC","TRBC2","CD2"],"CD8_T":["CD8A","CD8B","GZMK","GZMA","NKG7"],
    "CD4_T":["CD4","IL7R","CCR7","TCF7"],"Treg":["FOXP3","CTLA4","IL2RA","IKZF2"],
    "NK":["GNLY","NKG7","KLRD1","KLRF1","NCAM1"],"B_cell":["CD79A","CD79B","MS4A1","CD19","BANK1"],
    "Plasma":["MZB1","JCHAIN","IGHG1","XBP1","SDC1"],"Myeloid_Mono":["LYZ","CD14","S100A8","S100A9","FCN1"],
    "Macrophage":["CD68","CD163","C1QA","C1QB","APOE","MRC1"],"DC":["CLEC9A","FCER1A","CD1C","LAMP3","CLEC10A"],
    "pDC":["LILRA4","GZMB","IRF7","CLEC4C"],"Mast":["TPSAB1","TPSB2","CPA3","KIT","MS4A2"],
    "Neutrophil":["FCGR3B","CSF3R","CXCR2","G0S2"],"Fibroblast_CAF":["COL1A1","COL1A2","DCN","LUM","PDGFRB"],
    "Myofibroblast":["ACTA2","TAGLN","MYH11","RGS5"],"Endothelial":["PECAM1","VWF","CLDN5","CDH5","FLT1"],
    "Epithelial":["EPCAM","KRT8","KRT18","KRT19","CDH1"],"Squamous":["KRT5","KRT14","KRT6A","TP63","SFN"],
    "Hepatocyte":["ALB","APOA1","APOB","TTR"],"Neuro_NE":["PHOX2B","STMN2","SYP","CHGA","CHGB","ELAVL4"],
    "Photoreceptor":["RCVRN","CRX","RHO","PDE6B"],"Erythroid":["HBA1","HBA2","HBB","ALAS2"],
    "Proliferating":["MKI67","TOP2A","PCNA","CENPF"],
}

def build_lognorm(adata, counts_layer="counts"):
    import anndata as ad, scanpy as sc
    if counts_layer in adata.layers:
        X = adata.layers[counts_layer].copy()
    else:
        print("WARNING: layer not found; using .X as-is.")
        X = adata.X.copy()
    a = ad.AnnData(X=X, obs=adata.obs.copy(), var=adata.var.copy())
    a.obsm = adata.obsm.copy()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    return a

def top_markers(adata_lognorm, groupby="leiden", n=20, method="wilcoxon"):
    import scanpy as sc
    sc.tl.rank_genes_groups(adata_lognorm, groupby, method=method, use_raw=False)
    df = sc.get.rank_genes_groups_df(adata_lognorm, group=None)
    return df.groupby("group", observed=True).head(n).reset_index(drop=True)

def score_cell_types(adata_lognorm, marker_sets=MARKER_SETS, groupby="leiden"):
    import pandas as pd, scanpy as sc
    coverage, score_cols = {}, []
    for ct, genes in marker_sets.items():
        present = [g for g in genes if g in adata_lognorm.var_names]
        coverage[ct] = (len(present), len(genes))
        if present:
            col = "score_" + ct
            sc.tl.score_genes(adata_lognorm, present, score_name=col); score_cols.append(col)
    obs = adata_lognorm.obs
    mean_scores = obs.groupby(groupby, observed=True)[score_cols].mean()
    mean_scores.columns = [c.replace("score_","") for c in mean_scores.columns]
    z = (mean_scores - mean_scores.mean(axis=0)) / mean_scores.std(axis=0)
    suggested = z.idxmax(axis=1).rename("suggested_cell_type")
    coverage = pd.Series({k: str(v[0])+"/"+str(v[1]) for k,v in coverage.items()}, name="markers_found")
    return mean_scores, z, suggested, coverage


# ---- cell-type palette (defined once; reused across Fig 2/3) ----
CELLTYPE_ORDER = [
    "CD4+ T cell", "CD8+ T cell", "Regulatory T cell", "NK cell",
    "B cell", "Plasma cell",
    "Monocyte", "Macrophage", "Neutrophil/Granulocyte", "Mast cell",
    "Fibroblast/CAF", "Pericyte/vSMC", "Endothelial",
    "Epithelial/Malignant",
    "Neuronal (glioma)", "Neuronal (neuroblastoma)",
    "Erythroid", "Cycling cell",
]
CELLTYPE_COLORS = {
    "CD4+ T cell":            "#1f77b4",
    "CD8+ T cell":            "#4292c6",
    "Regulatory T cell":      "#9ecae1",
    "NK cell":                "#08519c",
    "B cell":                 "#6a51a3",
    "Plasma cell":            "#9e9ac8",
    "Monocyte":               "#fd8d3c",
    "Macrophage":             "#e6550d",
    "Neutrophil/Granulocyte": "#fdae6b",
    "Mast cell":              "#a63603",
    "Fibroblast/CAF":         "#238b45",
    "Pericyte/vSMC":          "#74c476",
    "Endothelial":            "#d62728",
    "Epithelial/Malignant":   "#525252",
    "Neuronal (glioma)":      "#c51b8a",
    "Neuronal (neuroblastoma)":"#f768a1",
    "Erythroid":              "#8c564b",
    "Cycling cell":           "#bcbd22",
}


# Shared lung normal pool: LUSC normal is too thin (351 cells) -> use LUAD+LUSC
# normal lung as the baseline for any LUSC tumour-vs-normal contrast.
# (Lambrechts et al., Nat Med 2018: tumours compared to matching non-malignant lung.)
NORMAL_TISSUE_GROUP = {
    "Lung Adenocarcinoma":          "Lung",
    "Lung Squamous Cell Carcinoma": "Lung",
}
def normal_pool_cancers(cancer):
    """Cancers whose Normal cells form the baseline for `cancer`."""
    grp = NORMAL_TISSUE_GROUP.get(cancer)
    if grp:
        return [c for c, g in NORMAL_TISSUE_GROUP.items() if g == grp]
    return [cancer]

TVN_EXCLUDE = ['Bladder Cancer', 'Lung Squamous Cell Carcinoma', 'Non-small Cell Lung Cancer', 'Papillary Thyroid Carcinoma', 'Squamous Cell Carcinoma in Situ', 'Distal Cholangiocarcinoma', 'Papillary Renal Cell Carcinoma']   # data-driven: cells<500 or samples<2


# ---- immune-subtype palette (Fig 3); grouped by compartment ----
IMMUNE_ORDER = [
    "CD4 naive","CD4 T memory","CD4 Tfh/CXCL13+","Regulatory T cell","T naive/resting",
    "CD8 cytotoxic","CD8 effector-memory","CD8 exhausted",
    "NK cytotoxic","NK CD56bright/tissue",
    "B naive","B memory","B cell","Plasma cell",
    "Monocyte","C1Q+ TAM","SPP1+ TAM","cDC","pDC","Neutrophil","Mast cell","Cycling",
]
IMMUNE_COLORS = {
    "CD4 naive":"#9ecae1","CD4 T memory":"#6baed6","CD4 Tfh/CXCL13+":"#3182bd",
    "Regulatory T cell":"#08519c","T naive/resting":"#c6dbef",
    "CD8 cytotoxic":"#74c476","CD8 effector-memory":"#41ab5d","CD8 exhausted":"#006d2c",
    "NK cytotoxic":"#1d9e75","NK CD56bright/tissue":"#66c2a4",
    "B naive":"#bcbddc","B memory":"#9e9ac8","B cell":"#dadaeb","Plasma cell":"#6a51a3",
    "Monocyte":"#fdae6b","C1Q+ TAM":"#e6550d","SPP1+ TAM":"#a63603","cDC":"#fdd0a2",
    "pDC":"#d94801","Neutrophil":"#fee391","Mast cell":"#993404","Cycling":"#969696",
}

def normal_eligible(cancer):
    """True if `cancer` may be used in tumour-vs-normal comparisons (>=2 normal donors)."""
    return cancer not in TVN_EXCLUDE
