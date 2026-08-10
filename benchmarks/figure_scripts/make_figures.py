import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

B = "/sessions/serene-loving-hawking/mnt/chisao/benchmarks"
OUT = "/sessions/serene-loving-hawking/mnt/CoWork/Papers/CS/ChiSao/figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9.5, "lines.linewidth": 1.8, "lines.markersize": 6,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3,
})

def expo(x, y):
    return np.polyfit(np.log(np.asarray(x,float)), np.log(np.asarray(y,float)), 1)[0]

def clean_logx(ax, ticks):
    ax.set_xscale("log"); ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([str(t) for t in ticks])

# ---------- Fig 1: same-hardware (CPU) wall-clock, log-log ----------
cpu = json.load(open(f"{B}/gpu_scaling_cpu.json"))
d = cpu["dims"]; ch = cpu["chisao"]["wall_s"]; bl = cpu["baselines_wall_s"]
fig, ax = plt.subplots(figsize=(5.6, 4.1))
ax.set_yscale("log")
ax.plot(d, ch, "o-", color="#1a5276", label=f"ChiSao (CPU)  $\\propto d^{{{expo(d,ch):.2f}}}$")
for name,(mk,col) in {"CMA-ES":("s--","#c0392b"),"DE":("^--","#27ae60"),"BasinHop":("d--","#8e44ad")}.items():
    w = bl[name]; ax.plot(d, w, mk, color=col, label=f"{name}  $\\propto d^{{{expo(d,w):.2f}}}$")
clean_logx(ax, d)
ax.annotate("$41$ s vs $277$ s\nat $d=512$", xy=(512, 277), xytext=(120, 150),
            fontsize=9.5, ha="center",
            arrowprops=dict(arrowstyle="->", lw=1, color="#333333"))
ax.set_xlabel("dimension $d$"); ax.set_ylabel("wall-clock time (s)")
ax.set_title("Same-hardware (CPU) wall-clock scaling", pad=10)
ax.legend(loc="lower right", framealpha=0.9)
fig.savefig(f"{OUT}/fig_walltime_loglog_cpu.png"); plt.close(fig)

# ---------- Fig 2: high-D GPU scaling to d=2048 ----------
hd = json.load(open(f"{B}/highdim_recovery_gpu.json"))["by_dim"]
dd = sorted(int(k) for k in hd)
wall = [hd[str(k)]["wall_s"] for k in dd]; fe = [hd[str(k)]["fes"] for k in dd]
fig, ax1 = plt.subplots(figsize=(6.0, 4.1))
c1, c2 = "#1a5276", "#c0392b"
ax1.set_yscale("log")
ax1.plot(dd, wall, "o-", color=c1, label=f"wall-clock  $\\propto d^{{{expo(dd,wall):.2f}}}$")
ax1.set_xlabel("dimension $d$"); ax1.set_ylabel("wall-clock time (s)", color=c1)
ax1.tick_params(axis="y", labelcolor=c1); clean_logx(ax1, dd)
ax1.set_xticklabels([str(t) for t in dd], rotation=45)
ax2 = ax1.twinx(); ax2.set_yscale("log")
ax2.plot(dd, fe, "s--", color=c2, label=f"function evals  $\\propto d^{{{expo(dd,fe):.2f}}}$")
ax2.set_ylabel("function evaluations", color=c2); ax2.tick_params(axis="y", labelcolor=c2); ax2.grid(False)
ax1.set_title("GPU scaling, shifted Rastrigin: 100% recovery, $L_\\infty\\approx0$, to $d=2048$", pad=10)
l1,la1 = ax1.get_legend_handles_labels(); l2,la2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, la1+la2, loc="upper left", framealpha=0.9)
fig.savefig(f"{OUT}/fig_highdim_scaling_gpu.png"); plt.close(fig)

print("wrote:")
for f in sorted(os.listdir(OUT)):
    print(" ", f, os.path.getsize(f"{OUT}/{f}"), "bytes")
