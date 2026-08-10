import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"serif","figure.dpi":200,"savefig.bbox":"tight"})
OUT="/sessions/serene-loving-hawking/mnt/CoWork/Papers/CS/ChiSao/figures/fig_flowchart_sixphase.png"
fig, ax = plt.subplots(figsize=(5.6, 7.4)); ax.axis("off")
ax.set_xlim(0,11); ax.set_ylim(5.3, 20.5)          # tight to content -> no empty band
def box(y, txt, color, x=5, w=6.4, h=1.25, fs=10):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.08",
                 fc=color, ec="#333333", lw=1.2))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs)
def arrow(y0, y1, x=5):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#333333"))
blue="#d6e4f0"; grey="#e8e8e8"; green="#d5f0d6"; amber="#f7e6c4"; red="#f2d0cf"
box(19,"Seed population on device\n(CarryTiger rays / random)",grey); arrow(18.35,17.9)
box(17.2,"Phase 1--2:  Converge\n(batched L-BFGS ascent)",blue); arrow(16.55,16.1)
box(15.4,"Stick-detect + freeze\n(grad-norm + quality gate)",green); arrow(14.75,14.3)
box(13.6,"Phase 3:  Deduplicate\n$L_\\infty$ hypercube hashing ($O(1)$)",blue); arrow(12.95,12.5)
box(11.8,"Phase 4:  Reseed\n(Repulse Monkey / Golden Rooster)",blue); arrow(11.15,10.7)
box(10.0,"Phase 5:  Hands Like Clouds\n(smoothed-$\\nabla f$ ascent)",amber); arrow(9.35,8.9)
box(8.2,"Phase 6:  Anti-converge\n(momentum $descent$, unfrozen)",red); arrow(7.55,7.0)
box(6.3,"Catalogue modes  $\\hat{M}^*$",grey,fs=11)
pc="#8e44ad"
ax.plot([8.2,9.6],[8.2,8.2],color=pc,lw=1.3)
ax.plot([9.6,9.6],[8.2,17.2],color=pc,lw=1.3)
ax.add_patch(FancyArrowPatch((9.6,17.2),(8.2,17.2),arrowstyle="-|>",mutation_scale=15,lw=1.3,color=pc))
ax.text(10.25,12.7,"oscillation $\\times\\,n_{\\mathrm{osc}}$",ha="center",va="center",fontsize=9,color=pc,rotation=90)
ax.add_patch(FancyArrowPatch((1.8,15.4),(1.8,6.3),connectionstyle="arc3,rad=0.5",
             arrowstyle="-|>",mutation_scale=15,lw=1.2,color="#27ae60",linestyle=(0,(4,2))))
ax.text(0.75,10.8,"frozen peaks",ha="center",va="center",fontsize=9,color="#27ae60",rotation=90)
ax.text(5,20.15,"ChiSao: fully device-resident six-phase cycle",ha="center",fontsize=11.5,weight="bold")
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.08); print("wrote", OUT)
