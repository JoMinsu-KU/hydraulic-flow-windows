"""Rebuild the paper figures from archived result summaries without reading signals."""
import argparse
import json
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[2] / "runs"
BLACK = "#171717"
GRAY = "#5b5b5b"
SINGLE = {"marker": "o", "markerfacecolor": "white", "markeredgecolor": BLACK,
          "markeredgewidth": 1.1, "markersize": 5.2, "linestyle": "none"}
PAIR = {"marker": "s", "markerfacecolor": BLACK, "markeredgecolor": BLACK,
        "markeredgewidth": .8, "markersize": 4.5, "linestyle": "none"}
PANEL_SYMBOLS = {
    "pilot": {"marker": "D", "s": 24, "facecolors": "white", "edgecolors": GRAY, "linewidths": .95},
    "confirm": {"marker": "o", "s": 12, "facecolors": BLACK, "edgecolors": BLACK, "linewidths": .45},
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.25,
    "axes.titlesize": 9, "axes.titleweight": "semibold", "axes.edgecolor": BLACK,
    "text.color": BLACK, "axes.labelcolor": BLACK, "xtick.color": BLACK, "ytick.color": BLACK,
    "axes.linewidth": .65, "xtick.major.width": .65, "ytick.major.width": .65,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6, "legend.fontsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False})


def load_results(run, filename="results.json"):
    return json.loads((RUNS / run / filename).read_text(encoding="utf8"))


def evaluated_points(record, boundary):
    # The binary-search log stores distances in raw-sum units. The event
    # radius is half a distance because the two closed intervals expand.
    scores = {Fraction(int(k), 1000): Fraction(v["oracle_group_then_window"]).limit_denominator(300)
              for k, v in record["by_bias_milli"].items()}
    for raw_distance, upper in boundary["binary_search_trace"].items():
        x = Fraction(int(raw_distance), 2 * boundary["scale"])
        y = Fraction(upper)
        assert x not in scores or scores[x] == y
        scores[x] = y
    for key in ("previous_event", "closed_threshold"):
        hit = boundary[key]
        assert scores[Fraction(hit["offset_fraction_l_min"])] == Fraction(hit["upper_fraction"])
    pts = sorted(scores.items())
    assert all(a[1] >= b[1] for a,b in zip(pts,pts[1:]))
    return np.array([(float(x), 100 * float(y)) for x,y in pts])


def equal_plateaus(ax, vals, color, linestyle):
    for (xa, ya), (xb, yb) in zip(vals, vals[1:]):
        if abs(ya - yb) < 1e-9:
            ax.hlines(ya, xa, xb, color=color, linewidth=.9, linestyles=linestyle, zorder=2)


def save(fig, name, folder):
    path = folder / (name + ".pdf")
    fig.savefig(path, metadata={"Creator": "Matplotlib", "Title": name, "CreationDate": None})
    plt.close(fig)


def draw_offset(folder):
    results = {name: (load_results(run), load_results(run, "critical-offset.json"))
               for name, run in [("pilot", "u-pilot-02"), ("confirm", "u-confirm-01")]}
    pts = {name: evaluated_points(*data) for name,data in results.items()}
    fig, (wide, zoom) = plt.subplots(1,2,figsize=(7.03,2.64),width_ratios=(1.07,1))
    fig.subplots_adjust(left=.079,right=.982,bottom=.20,top=.79,wspace=.31)
    for ax in (wide,zoom):
        ax.set_xlabel("Common offset half-width, $B$ (L/min)")
        ax.axhline(85, color=GRAY, lw=.9, linestyle=(0,(3,2)), zorder=2)
        ax.spines["left"].set_position(("outward", 2))
        ax.tick_params(pad=2)
    wide.set_title("(a)  Evaluated offset radii",loc="left",pad=6)
    zoom.set_title("(b)  First crossing below 85%",loc="left",pad=6)
    wide.set_ylabel("Balanced accuracy (%)", labelpad=4)
    wide.set(xlim=(-.007,.309), ylim=(32,103), xticks=(0,.05,.10,.15,.20,.25,.30),
             yticks=(40,55,70,85,100))
    wide.xaxis.set_major_formatter(FuncFormatter(lambda x,_: "0" if x==0 else f"{x:.2f}"))
    for (xa,ya),(xb,yb) in zip(pts["confirm"],pts["confirm"][1:]):
        if ya>yb:
            wide.add_patch(Rectangle((xa,yb),xb-xa,ya-yb,facecolor=".90",edgecolor="none",zorder=0))
    wide.axhline(100*results["confirm"][0]["two_mean_score"], color=BLACK, lw=1.05, zorder=1)
    wide.text(.304,90.5,"Two means, held: 87%",ha="right",va="bottom",fontsize=7.4)
    wide.annotate("39%", xy=(.30,39), xytext=(.302,46), ha="right",fontsize=7.5)
    for name in ("pilot","confirm"):
        plotted = pts[name][pts[name][:,0] <= .30]
        equal_plateaus(wide,plotted,GRAY if name=="pilot" else BLACK,"dashed" if name=="pilot" else "solid")
        wide.scatter(*plotted.T, **PANEL_SYMBOLS[name],zorder=4)
    zoom.set(xlim=(.05868,.05949),ylim=(83.9,86.25), yticks=(84,84.5,85,85.5,86),
             xticks=(.0588,.0590,.0592,.0594))
    zoom.xaxis.set_major_formatter(FuncFormatter(lambda val,_: f"{val:.4f}"))
    zoom.yaxis.set_major_formatter(FuncFormatter(lambda val,_: f"{val:g}"))
    zoom.tick_params(pad=2)
    for name in ("pilot","confirm"):
        pair = results[name][1]
        left,right=(pair[k] for k in ("previous_event","closed_threshold"))
        x0,x1=left["offset_l_min"],right["offset_l_min"]
        y0,y1=100*left["upper"],100*right["upper"]
        inside=pts[name][(pts[name][:,0] >= .05868) & (pts[name][:,0] <= .05949)]
        c=GRAY if name=="pilot" else BLACK
        equal_plateaus(zoom,inside,c,"dashed" if name=="pilot" else "solid")
        zoom.plot([x0,x1,x1],[y0,y0,y1],color=c,lw=1,zorder=3)
        zoom.scatter(*inside.T,**PANEL_SYMBOLS[name],zorder=4)
        xtext,ytext=(.05889,84.11) if name=="pilot" else (.05922,85.95)
        zoom.annotate(f"{x1:.5f}",(x1,y1),(xtext,ytext),ha="center", va="center",fontsize=7.5,
                      arrowprops={"arrowstyle":"-","lw":.65,"color":c,"shrinkB":3})
    legend_items=[Line2D([],[],marker="D",linestyle="none",ms=4.5,mfc="white",mec=GRAY,
                         label="One mean: pilot upper bound"),
                  Line2D([],[],marker="o",linestyle="none",ms=3.5,mfc=BLACK,mec=BLACK,
                         label="One mean: held upper bound"),
                  Line2D([],[],color=GRAY,linestyle=(0,(3,2)),lw=.9,label="85% target")]
    fig.legend(handles=legend_items,ncol=3,loc="upper center",bbox_to_anchor=(.52,.996),
               frameon=False,handletextpad=.45,columnspacing=1.35)
    save(fig,"offset-bound",folder)


def dot_key(fig, labels):
    handles=[Line2D([],[],**style,label=label) for style,label in zip((SINGLE,PAIR),labels)]
    fig.legend(handles=handles,ncol=2,frameon=False,loc="upper center",bbox_to_anchor=(.50,.998),
               handletextpad=.28,columnspacing=1.0)


def dot_rows(ax, first, second):
    y=np.arange(len(first))
    for yl, p1, p2 in zip(y,first,second):
        ax.plot([p1,p2],[yl,yl],color=".55",lw=1.05,zorder=1)
    ax.plot(first,y,**SINGLE,zorder=3)
    ax.plot(second,y,**PAIR,zorder=3)
    ax.invert_yaxis()
    ax.tick_params(axis="y",length=0,pad=4)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x",linestyle=(0,(1,3)),color=".72",lw=.55)
    ax.set_axisbelow(True)


def draw_held(folder):
    data=load_results("u-confirm-01")
    bounded=data["by_bias_milli"]["300"]["oracle_upper_by_group"]
    attained=data["two_mean_by_group"]
    assert len(bounded)==len(attained)==len(data["groups"])==10
    assert sum(a>b for a,b in zip(attained,bounded))==9
    fig,ax=plt.subplots(figsize=(3.45,3.09))
    fig.subplots_adjust(left=.29,right=.976,bottom=.143,top=.796)
    ax.axvline(85,color=GRAY,lw=.75,ls=(0,(3,2)))
    dot_rows(ax,np.array(bounded)*100,np.array(attained)*100)
    group_labels=["  /  ".join(str(v) for v in setting) for setting in data["groups"]]
    ax.set(yticks=np.arange(10),yticklabels=group_labels,xticks=[30,45,60,75,90,100],
           xlim=(28,102),xlabel="Balanced accuracy (%)",ylim=(9.5,-.5))
    fig.text(.026,.832,"Setting (C / V / A)",fontsize=7.0,va="center")
    dot_key(fig,["One-mean upper bound","Two-mean rule"])
    save(fig,"held-conditions",folder)


def draw_acquisition(folder):
    acq=load_results("u-paper-wave-acquisition","result.json")["results"]
    grid=load_results("u-wave-gain-region")["grid"]
    joint={}
    for label in ("single","pair"):
        cell=next(c for c in grid if c["layout"]==label and c["gain_ppm"]==2500 and c["error_micro"]==5000)
        assert cell["unknown_n"]==0 and cell["lower_n"]==cell["upper_n"]
        joint[label]=cell["lower_n"]
    labels=["Aligned, clean","Gain + window error","Unknown phase","Known phase, refit","Zero floor + offset"]
    def get(layout):
        fixed=acq[layout]
        n=[fixed["0"]["fixed"],joint["single" if layout=="one" else "pair"],
           fixed["all_shifts"]["fixed"],fixed["all_shifts"]["known_phase"],fixed["0"]["0.3"]["pass"]]
        return np.asarray(n)*100/300
    one,two=get("one"),get("two")
    fig,ax=plt.subplots(figsize=(3.45,2.56))
    fig.subplots_adjust(left=.39,right=.977,bottom=.19,top=.824)
    ax.axvline(85,color=GRAY,lw=.75,ls=(0,(3,2)))
    dot_rows(ax,one,two)
    ax.set(yticks=np.arange(len(labels)),yticklabels=labels,xlim=(27,104),xticks=[30,45,60,75,90,100],
           xlabel="Balanced accuracy (%)",ylim=(4.52,-.5))
    dot_key(fig,["One window, 50 samples","Two windows, 25 + 25"])
    save(fig,"acquisition-limits",folder)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=HERE)
    output=parser.parse_args().out
    output.mkdir(parents=True,exist_ok=True)
    draw_offset(output)
    draw_held(output)
    draw_acquisition(output)
