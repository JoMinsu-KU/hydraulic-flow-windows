"""Plot previously tabulated, grade-specific ZeMA pilot summaries."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = HERE.parents[2] / "runs"
WINDOWS = ("first", "late")
STYLES = {
    "first": dict(marker="o", markersize=4.8, markerfacecolor="white",
                  markeredgecolor=".10", markeredgewidth=1.15),
    "late": dict(marker="s", markersize=4.5, markerfacecolor=".15",
                 markeredgecolor=".15", markeredgewidth=.8),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def draw(runs, output):
    study = read_json(runs / "u-paper-applicability" / "result.json")
    timing = read_json(runs / "u-paper-window-registration-diagnostic" / "results.json")
    grades = [str(c) for c in (0, 1, 2)]
    stats = study["phase"]["FS1_L_min"]
    shifts = {str(r["grade"]): r for r in timing["by_grade"]}
    assert study["rows"]["pilot"] == 300 and all(shifts[g]["n"] == 100 for g in grades)
    assert all(shifts[g]["negative_zero_positive_delta_n"] == [100, 0, 0] for g in grades)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.3,
        "xtick.labelsize": 7.8, "ytick.labelsize": 8, "axes.titlesize": 9,
        "axes.titleweight": "semibold", "text.color": ".09", "axes.labelcolor": ".09",
        "axes.edgecolor": ".12", "axes.linewidth": .65, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False, "pdf.fonttype": 42,
        "xtick.major.size": 3, "xtick.major.width": .65, "ytick.major.size": 0,
    })
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(7.03, 2.06),
                             gridspec_kw={"width_ratios": (1, 1, 1.13)})
    fig.subplots_adjust(left=.081, right=.978, bottom=.239, top=.76, wspace=.27)
    y = np.arange(len(grades))
    for ax, window, title in zip(axes, WINDOWS, ("(a)  0–5 s", "(b)  50–55 s")):
        quantiles = np.array([[stats[g][window][k] for k in ("p05", "median", "p95")] for g in grades])
        assert (np.diff(quantiles, axis=1) >= 0).all()
        lo, med, hi = quantiles.T
        ax.hlines(y, lo, hi, color=".25", linewidth=1.35)
        ax.vlines(lo, y-.095, y+.095, color=".25", linewidth=.75)
        ax.vlines(hi, y-.095, y+.095, color=".25", linewidth=.75)
        ax.plot(med, y, linestyle="none", zorder=3, **STYLES[window])
        ax.set_title(title, loc="left", pad=5)
        ax.set_xlabel("Mean flow (L/min)",labelpad=3)
    axes[0].set(xlim=(.172, .218), xticks=(.18,.19,.20,.21))
    axes[0].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[1].set(xlim=(7.33,7.92), xticks=(7.4,7.6,7.8))
    ax = axes[2]
    ax.set_title("(c)  One-sample delay", loc="left", pad=5)
    ax.axvline(0, linewidth=.7, linestyle=(0,(3,2)), color=".50")
    for window, key in zip(WINDOWS, ("median_first_mean_change_l_min", "median_last_mean_change_l_min")):
        change = [shifts[g][key] for g in grades]
        ax.plot(change, y, linestyle="none", **STYLES[window])
    ax.set(xlim=(-.180,.012), xticks=(-.16,-.08,0), xlabel="Median change in mean (L/min)")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    for ax in axes:
        ax.grid(axis="x", color=".78", linewidth=.55, linestyle=(0,(1,3)))
        ax.set_axisbelow(True)
        ax.tick_params(axis="y",pad=4)
    axes[0].set(yticks=y, yticklabels=("None","Weak","Severe"), ylim=(2.45,-.45))
    items = [Line2D([],[],linestyle="none",label=l,**STYLES[w]) for w,l in
             zip(WINDOWS, ("First window, 0–5 s","Later window, 50–55 s"))]
    fig.legend(handles=items, ncol=2, loc="upper center", bbox_to_anchor=(.50,1.001),
               fontsize=7.7, frameon=False, handletextpad=.3,columnspacing=1.45)
    fig.savefig(output, metadata={"Title": "Nominal flow contrast and the effect of a one-sample delay",
                                  "Creator": "Matplotlib", "CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS,
                        help="Directory containing the two archived experiment folders")
    parser.add_argument("--out", type=Path, default=HERE / "phase-evidence.pdf")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    draw(args.runs, args.out)
