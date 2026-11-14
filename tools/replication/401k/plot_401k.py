"""Plot the worked architecture in Table 11 of Section 6.5. Input: the replication estimates CSV.

Usage: python plot_401k.py aipw_estimates.csv eligibility-401k.png
Requires matplotlib. No household data are needed.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

names = ("G", "GL", "GLT", "Unpenalized", "Dropout", "L1")
labels = ("G-BNN", "GL-BNN", "GLT-BNN", "Unpenalized", "Dropout", r"$L_1$ penalty")
with Path(sys.argv[1]).open(newline="", encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f) if (int(r["depth"]), int(r["width"])) == (10, 25) and float(r["bound"]) == 0.01]
assert len(rows) == 6 and {r["method"] for r in rows} == set(names)
data = {r["method"]: [float(r[k]) for k in ("ate", "ci_low", "ci_high")] for r in rows}
fig = plt.figure(figsize=(10.4, 4.65), facecolor="white")
ink, muted, accent, grey = "#172d3a", "#415665", "#006b64", "#cbd6db"
fig.text(.03,.91,"401(k) eligibility and household net financial assets",color=ink,weight="bold",size=19)
fig.text(.03,.837,"Five-fold adjusted differences for 9,915 households  ·  10 hidden layers of 25 units",color=muted,size=12)
ax=fig.add_axes([.205,.165,.55,.575])
ax.set(xlim=(0,15000),ylim=(-.55,5.55))
ax.set_axisbelow(True)
ax.axhspan(2.60,3.40,color="#eaf5f2",lw=0,zorder=0)
ax.xaxis.set_major_locator(MultipleLocator(3000))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: f"{int(x):,}"))
ax.grid(axis="x",color="#e0e7e9",lw=.8)
ax.axvline(0,color="#95a5aa",lw=1.)
ax.set_yticks(range(5,-1,-1),labels)
ax.tick_params(axis="y",length=0,pad=12,labelcolor=ink,labelsize=12)
ax.tick_params(axis="x",length=3,color=grey,labelcolor=muted,labelsize=10)
for edge in ax.spines.values(): edge.set_visible(False)
ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(grey)
ax.set_xlabel("Adjusted difference (1991 dollars)",size=11,labelpad=7,color=muted)
ax.text(1.035,1.03,"Estimate  [95% CI]",transform=ax.transAxes,size=11,color=muted,weight="bold")
for j,method in enumerate(names):
    val,lo,hi=data[method]; yy=5-j; key=method=="GLT"; color=accent if key else "#4e6671"
    assert lo<val<hi
    ax.errorbar(val,yy,xerr=[[val-lo],[hi-val]],fmt="D" if key else "o",color=color,markersize=8 if key else 6,capsize=4,elinewidth=2.0 if key else 1.45,zorder=3)
    ax.text(1.035,yy,f"{val:,.0f}  [{lo:,.0f}, {hi:,.0f}]",transform=ax.get_yaxis_transform(),va="center",color=color,weight="bold" if key else "normal",size=11)
    if key:
        lab=ax.get_yticklabels()[j];lab.set_weight("bold");lab.set_color(accent)
fig.savefig(sys.argv[2],dpi=165,facecolor="white",metadata={"Software":"Matplotlib"})
