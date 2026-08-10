import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
B="/sessions/serene-loving-hawking/mnt/chisao/benchmarks"
OUT="/sessions/serene-loving-hawking/mnt/CoWork/Papers/CS/ChiSao/figures/fig_walltime_scaling.png"
cpu=json.load(open(f"{B}/gpu_scaling_chisao_cpu_full.json"))
gpu=json.load(open(f"{B}/gpu_scaling_chisao_gpu_full.json"))
low=json.load(open(f"{B}/gpu_scaling_ackley_lowd.json"))
cap=json.load(open(f"{B}/gpu_scaling_baselines_capped.json"))
def expo(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=~np.isnan(y)
    return np.polyfit(np.log(x[m]),np.log(y[m]),1)[0] if m.sum()>=2 else float('nan')
plt.rcParams.update({"font.family":"serif","font.size":11,"axes.titlesize":12,"axes.labelsize":12,
 "xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9,"lines.linewidth":1.9,
 "figure.dpi":200,"savefig.bbox":"tight","axes.grid":True,"grid.alpha":0.3})
fig,ax=plt.subplots(figsize=(8.2,5.2)); ax.set_xscale("log"); ax.set_yscale("log")
def series(D,W,R,mk,col,label,solid):
    D=np.array(D,float); W=np.array(W,float); R=np.array(R,bool)
    ax.plot(D,W,ls=("-" if solid else "--"),color=col,lw=1.8,zorder=2)
    fm=R&~np.isnan(W); hm=(~R)&~np.isnan(W)
    ax.plot(D[fm],W[fm],mk,mfc=col,mec=col,ms=6.5,ls="none",zorder=3)
    ax.plot(D[hm],W[hm],mk,mfc="white",mec=col,ms=6.5,mew=1.4,ls="none",zorder=3)
    ax.plot([],[],mk,color=col,ls=("-" if solid else "--"),label=label)
cd=cpu["dims"]; cw=cpu["chisao"]["wall_s"]
series(cd,cw,[True]*len(cd),"s","#2e86c1",f"ChiSao, CPU  ($\\propto d^{{{expo(cd,cw):.2f}}}$)",True)
gd=gpu["dims"]; gw=gpu["chisao"]["wall_s"]
series(gd,gw,[True]*len(gd),"o","#154360",f"ChiSao, GPU  ($\\propto d^{{{expo(gd,gw):.2f}}}$)",True)
def merge(nm):
    w=list(low["baselines_wall_s"][nm][:5])
    cc=[x if x is not None else np.nan for x in cap["baselines_wall_s"][nm]]
    return [2,4,8,16,32,64,128,256,512,1024,2048], w+cc
for nm,mk,col,lbl in [("CMA-ES","^","#c0392b","CMA-ES, CPU"),("DE","v","#27ae60","DE, CPU"),
                      ("BasinHop","D","#8e44ad","Basin-hopping, CPU")]:
    D,W=merge(nm); series(D,W,[d<=8 for d in D],mk,col,f"{lbl}  ($\\propto d^{{{expo(D,W):.2f}}}$)",False)
budget=cw[-1]
ax.axhline(budget,color="0.45",ls=(0,(6,3)),lw=1.3)
ax.text(2.1,budget*1.16,f"$\\approx${budget:.0f} s: ChiSao CPU solves $d{{=}}2048$. Same budget: CMA-ES $\\to$256, DE $\\to$512, BH $\\to$1024 (all 0% recovery)",
        fontsize=7.6,color="0.3",va="bottom")
alld=[2,4,8,16,32,64,128,256,512,1024,2048]
ax.set_xticks(alld); ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticklabels([str(d) for d in alld],rotation=45)
ax.set_xlabel("dimension $d$"); ax.set_ylabel("mean wall-clock time (s)")
ax.set_title("Wall-clock scaling on Ackley ($d=2$ to $2048$)")
ax.plot([],[],"o",mfc="0.3",mec="0.3",ls="none",label="filled: recovered")
ax.plot([],[],"o",mfc="white",mec="0.3",ls="none",label="hollow: 0% recovery")
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.15),ncol=3,framealpha=0.95,columnspacing=1.2,fontsize=8.3)
fig.savefig(OUT); print("wrote",OUT,"| budget=%.0f | cpu_exp=%.2f gpu_exp=%.2f"%(budget,expo(cd,cw),expo(gd,gw)))
