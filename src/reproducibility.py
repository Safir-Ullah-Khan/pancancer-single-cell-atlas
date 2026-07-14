# =====================================================================
# reproducibility.py — Reproducibility, robustness & resource (Supplementary Fig. 5)
# Final manuscript figure: Supplementary Figure 5 (see FIGURE_MAP.md)
# (internal working name: Figure 8)
# FIGURE 8 — full render (7 panels a–g): reproducibility, robustness & resource
# Nature spec: 183mm, 600dpi, vector PDF. save_panel bypasses bbox override.
# =====================================================================
import sys, os, shutil, json
sys.path.insert(0,"/data/notebooks")
import numpy as np, pandas as pd, matplotlib as mpl
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
from atlas_setup import set_nature_style, mm
set_nature_style()
mpl.rcParams.update({
    "font.size":9,"axes.titlesize":13,"axes.labelsize":11,"xtick.labelsize":9,"ytick.labelsize":9,
    "legend.fontsize":9,"pdf.fonttype":42,"font.family":"sans-serif","font.weight":"medium",
    "text.color":"#000","axes.labelcolor":"#000","xtick.color":"#000","ytick.color":"#000","axes.titlecolor":"#000",
    "savefig.bbox":"standard","savefig.pad_inches":0.05,
})
OUT8="/data/paper_tumoronly/Figure8"; PAN=f"{OUT8}/panels"; RB=f"{OUT8}/robustness"; CO=f"{OUT8}/concordance"
os.makedirs(PAN,exist_ok=True); DL="/home/ubuntu/downloads_tumoronly"; os.makedirs(DL,exist_ok=True)
TT,LB,TK=13,11.5,10.5
def save_panel(fig,path):
    fig.savefig(path+".png",dpi=600,bbox_inches="tight",pad_inches=0.06)
    fig.savefig(path+".pdf",bbox_inches="tight",pad_inches=0.06); plt.close(fig)
def style_axis(ax,xlabel=None,ylabel=None,title=None):
    if title: ax.set_title(title,loc="left",fontsize=TT,fontweight="bold",color="#000")
    if xlabel: ax.set_xlabel(xlabel,fontsize=LB,fontweight="bold",color="#000")
    if ylabel: ax.set_ylabel(ylabel,fontsize=LB,fontweight="bold",color="#000")
    for lab in ax.get_xticklabels()+ax.get_yticklabels(): lab.set_color("#000"); lab.set_fontweight("medium")
    ax.tick_params(colors="#000",width=1.0)
# ---- load data ----
ds=pd.read_csv(f"{RB}/downsampling.csv")
sh=pd.read_csv(f"{RB}/split_half.csv")
ps=pd.read_csv(f"{RB}/parameter_sensitivity.csv")
spec=pd.read_csv(f"{RB}/program_specificity.csv", index_col=0)
conc=pd.read_csv(f"{CO}/gavish_concordance.csv")
summ=json.load(open(f"{RB}/resource_summary.json"))
# ===== 8a schematic =====
fig,ax=plt.subplots(figsize=mm(183,44)); ax.set_xlim(0,100);ax.set_ylim(-2,26);ax.axis("off")
ax.add_patch(FancyBboxPatch((1,2),98,22,boxstyle="round,pad=0.5,rounding_size=2",fc="#fafafc",ec="none",zorder=0))
stages=[(17,"Robustness","#238b45","downsampling · split-half"),
        (50,"Validation","#2c7fb8","literature concordance"),
        (83,"Resource","#E69F00","4.83M-cell atlas")]
for cx,title,col,sub in stages:
    ax.add_patch(FancyBboxPatch((cx-14,7),28,12,boxstyle="round,pad=0.3,rounding_size=1.6",fc="white",ec=col,lw=2,zorder=2))
    ax.text(cx,15,title,ha="center",fontsize=11,fontweight="bold",color=col,zorder=3)
    ax.text(cx,10,sub,ha="center",fontsize=7,color="#000",zorder=3)
for x0 in [31,64]:
    ax.add_patch(FancyArrowPatch((x0,13),(x0+5,13),arrowstyle="-|>",mutation_scale=14,lw=2,color="#888",zorder=2))
ax.text(50,23,"Reproducibility & community resource",ha="center",fontsize=10,fontweight="bold",zorder=4,path_effects=[pe.withStroke(linewidth=2.5,foreground="white")])
save_panel(fig,f"{PAN}/Fig8a_schematic")
# ===== 8b downsampling =====
fig,ax=plt.subplots(figsize=mm(85,66))
ax.errorbar(ds["fraction"]*100, ds["mean_r"], yerr=ds["sd_r"], marker="o", ms=7, lw=2, color="#238b45", capsize=3)
ax.set_ylim(0.98,1.001); ax.set_xlabel("Cells sampled (%)",fontsize=LB,fontweight="bold")
style_axis(ax, ylabel="Correlation to full", title="Downsampling robustness")
save_panel(fig,f"{PAN}/Fig8b_downsampling")
# ===== 8c split-half =====
sh=sh.sort_values("split_half_r")
fig,ax=plt.subplots(figsize=mm(85,66))
ax.barh(range(len(sh)), sh["split_half_r"], color="#2c7fb8", height=0.7)
ax.set_xlim(0.95,1.002); ax.set_yticks(range(len(sh))); ax.set_yticklabels(sh["state"],fontsize=9)
style_axis(ax, xlabel="Split-half correlation", title="Reproducibility")
save_panel(fig,f"{PAN}/Fig8c_splithalf")
# ===== 8d Gavish concordance =====
best=conc.loc[conc.groupby("our_program")["jaccard"].idxmax()].copy()
best["neglogp"]=-np.log10(best["pval"]+1e-30)
best=best.sort_values("neglogp")
fig,ax=plt.subplots(figsize=mm(90,66))
ax.barh(range(len(best)), best["neglogp"], color="#6a51a3", height=0.7)
ax.set_yticks(range(len(best))); ax.set_yticklabels([f"{r.our_program}→{r.gavish_MP.replace('MP_','')}" for r in best.itertuples()],fontsize=7.5)
ax.axvline(2,ls="--",lw=0.8,color="#333"); ax.text(2.1,0.3,"p<0.01",fontsize=7,color="#333")
style_axis(ax, xlabel="−log10 p (overlap)", title="Concordance with published meta-programs")
save_panel(fig,f"{PAN}/Fig8d_concordance")
# ===== 8e parameter sensitivity =====
fig,ax=plt.subplots(figsize=mm(85,60))
ax.bar(range(len(ps)), ps["correlation_to_standard"], color="#E69F00", edgecolor="#333", linewidth=0.4)
ax.set_ylim(0.9,1.005); ax.set_xticks(range(len(ps)))
ax.set_xticklabels([p.replace("target_sum=","ts=").replace("ctrl_size=","cs=") for p in ps["parameter"]],rotation=30,ha="right",fontsize=8)
style_axis(ax, ylabel="Correlation to standard", title="Parameter sensitivity")
save_panel(fig,f"{PAN}/Fig8e_paramsens")
# ===== 8f compartment specificity =====
fig,ax=plt.subplots(figsize=mm(95,68))
progs=spec.columns.tolist(); comps=spec.index.tolist()
x=np.arange(len(progs)); w=0.26
comp_col={"Malignant":"#B2182B","Immune":"#2c7fb8","Stromal":"#238b45"}
for i,c in enumerate(comps):
    ax.bar(x+(i-1)*w, spec.loc[c], w, label=c, color=comp_col.get(c,"#888"), edgecolor="#333", linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(progs, rotation=90, fontsize=8)
ax.axhline(0,color="#333",lw=0.6)
ax.set_ylim(spec.values.min()-0.1, spec.values.max()+0.25)
leg=ax.legend(fontsize=7.5,frameon=True,ncol=1,loc="upper left",bbox_to_anchor=(0.01,0.99),
              handletextpad=0.4,borderpad=0.4,framealpha=0.9,edgecolor="none")
style_axis(ax, ylabel="Program score", title="Program compartment specificity")
save_panel(fig,f"{PAN}/Fig8f_specificity")
# ===== 8g resource infographic =====
fig,ax=plt.subplots(figsize=mm(90,66)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
ax.set_title("Pan-cancer atlas resource",loc="left",fontsize=TT,fontweight="bold")
stats=[("4.83M","cells","#6a51a3"),("30","cancers","#B2182B"),("1,010","samples","#2c7fb8"),
       ("89,530","malignant cells","#238b45"),("17","cell types","#E69F00"),("17","TCGA cohorts","#d62728")]
for i,(num,lab,col) in enumerate(stats):
    r,c=divmod(i,3); cx=1.7+c*3.3; cy=6.5-r*3.6
    ax.add_patch(FancyBboxPatch((cx-1.4,cy-1.3),2.8,2.5,boxstyle="round,pad=0.05,rounding_size=0.3",fc=col,alpha=0.12,ec=col,lw=1.5))
    ax.text(cx,cy+0.35,num,ha="center",fontsize=13,fontweight="bold",color=col)
    ax.text(cx,cy-0.7,lab,ha="center",fontsize=7.5,color="#000")
save_panel(fig,f"{PAN}/Fig8g_resource")
# ========== COMPOSITE (7 panels) ==========
panels={k:f"{PAN}/Fig8{k}_{n}.png" for k,n in
        [("a","schematic"),("b","downsampling"),("c","splithalf"),("d","concordance"),
         ("e","paramsens"),("f","specificity"),("g","resource")]}
fig=plt.figure(figsize=mm(183,215))
gs=fig.add_gridspec(4,2,height_ratios=[0.36,1.0,1.0,1.0],hspace=0.20,wspace=0.16)
slots={"a":gs[0,:],"b":gs[1,0],"c":gs[1,1],"d":gs[2,0],"e":gs[2,1],"f":gs[3,0],"g":gs[3,1]}
for lab,sl in slots.items():
    a=fig.add_subplot(sl);a.axis("off");a.imshow(mpimg.imread(panels[lab]))
    a.text(-0.01,1.02,lab,transform=a.transAxes,fontsize=15,fontweight="bold",va="bottom",ha="right",color="#000")
fig.savefig(f"{OUT8}/Figure8_full.png",dpi=600,bbox_inches="tight")
fig.savefig(f"{OUT8}/Figure8_full.pdf",bbox_inches="tight")
for ext in (".png",".pdf"): shutil.copy(f"{OUT8}/Figure8_full{ext}",DL)
print("saved Figure8_full")
