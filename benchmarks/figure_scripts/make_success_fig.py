import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
OUT="/sessions/serene-loving-hawking/mnt/CoWork/Papers/CS/ChiSao/figures/fig_success_rate.png"
dims=[2,8,32,64]
# recovery (%) per function, order [d2,d8,d32,d64], transcribed from Tables 3-4 (Groups A+B)
data={
 "chisao_rnd":[[100,100,100,100],[100,0,0,0],[100,0,0,0],[10,100,100,100],[100,100,100,100],[100,100,100,100],[100,100,100,100],
               [100,100,100,100],[20,100,100,100],[100,100,100,100],[100,100,100,100],[100,100,10,0],[100,100,100,100]],
 "chisao_ct":[[100,100,100,100],[100,100,100,100],[100,10,0,0],[100,100,100,100],[100,100,100,100],[100,100,100,100],[100,100,100,100],
              [100,100,100,100],[50,100,100,100],[100,100,100,100],[100,100,100,100],[100,100,30,0],[100,100,100,100]],
 "de":[[90,0,0,0],[100,100,0,0],[90,80,0,0],[60,0,80,80],[100,100,0,0],[100,100,0,0],[100,100,100,100],
       [100,100,100,100],[100,80,90,40],[100,100,100,100],[100,100,100,100],[100,100,100,100],[100,100,100,100]],
 "bh":[[60,0,0,0],[100,20,0,0],[0,0,0,0],[0,0,90,100],[100,100,40,0],[40,0,0,0],[100,100,100,100],
       [100,100,100,100],[100,90,100,100],[100,100,100,100],[100,100,100,100],[100,100,100,100],[100,100,100,100]],
 "cma":[[70,0,0,0],[100,100,0,0],[80,0,0,0],[50,30,0,0],[100,100,0,0],[90,70,0,0],[100,100,100,100],
        [100,100,0,0],[100,0,0,0],[100,100,0,0],[100,100,100,100],[100,100,100,100],[100,100,0,0]],
}
means={k:np.mean(np.array(v),axis=0) for k,v in data.items()}
plt.rcParams.update({"font.family":"serif","font.size":11,"axes.titlesize":12,"axes.labelsize":12,
    "xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9.5,"lines.linewidth":1.9,
    "lines.markersize":6,"figure.dpi":150,"savefig.bbox":"tight","axes.grid":True,"grid.alpha":0.3})
fig,ax=plt.subplots(figsize=(5.8,4.1))
ax.set_xscale("log")
ax.plot(dims,means["chisao_ct"],"o-",color="#1a5276",label="ChiSao (carry_tiger)")
ax.plot(dims,means["chisao_rnd"],"s-",color="#2e86c1",label="ChiSao (random)")
ax.plot(dims,means["cma"],"^--",color="#c0392b",label="CMA-ES")
ax.plot(dims,means["de"],"v--",color="#27ae60",label="DE")
ax.plot(dims,means["bh"],"d--",color="#8e44ad",label="Basin-hopping")
ax.set_xticks(dims); ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticklabels([str(d) for d in dims])
ax.set_xlabel("dimension $d$"); ax.set_ylabel("mean mode-recovery rate (%)")
ax.set_ylim(-3,103)
ax.set_title("Mean recovery vs. dimension (Groups A+B, 13 functions)")
ax.legend(loc="lower left", framealpha=0.9)
fig.savefig(OUT); print("wrote",OUT)
for k in means: print(f"  {k:12s}", " ".join(f"{x:5.1f}" for x in means[k]))
