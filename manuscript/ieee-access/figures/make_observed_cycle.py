"""Draw one observed ZeMA cycle from the public data.zip or its UCI wrapper.

From the repository root:
    python manuscript/ieee-access/figures/make_observed_cycle.py --archive /path/to/data.zip

The default output is figures/observed-cycle.pdf. Use --out to select another
location. FS1 and PS2 are plotted at their documented sampling rates. Selection
uses the released split before inspecting waveform values or predictions.
"""
import argparse
import csv
import json
from contextlib import ExitStack
from decimal import Decimal
from hashlib import sha256
from io import BytesIO
from itertools import islice
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
WINDOWS = ((0, 5), (50, 55))


def named_member(archive, basename):
    found = [info for info in archive.infolist()
             if PurePosixPath(info.filename).name.lower() == basename.lower()]
    if len(found) != 1:
        raise ValueError(f"Expected one {basename} member, found {len(found)}")
    return found[0]


def read_at(archive, name, index, size):
    with archive.open(named_member(archive, name)) as stream:
        tokens = next(islice(stream, index, index + 1)).split()
    if len(tokens) != size:
        raise ValueError(f"{name}: expected {size} values in row {index}, found {len(tokens)}")
    return [Decimal(v.decode("ascii")) for v in tokens]


def reference_row(root, relative, row_key, index):
    with (root / relative).open(newline="", encoding="utf-8-sig") as stream:
        return next(row for row in csv.DictReader(stream) if int(row[row_key]) == index)


def load_selected(path, root):
    with (root / "cycle_split.csv").open(newline="", encoding="utf-8-sig") as fp:
        eligible = [r for r in csv.DictReader(fp) if r["role"] == "pilot" and r["pump"] == "0"
                    and r["stable_flag"] == "0"]
    entry = min(eligible, key=lambda row: int(row["cycle_zero_based"]))
    index = int(entry["cycle_zero_based"])
    with ExitStack() as stack:
        archive = stack.enter_context(ZipFile(path))
        if not any(PurePosixPath(n).name.lower() == "fs1.txt" for n in archive.namelist()):
            inner = named_member(archive, "data.zip")
            if inner.file_size > 128 * 1024**2:
                raise ValueError("Unexpected inner archive size")
            archive = stack.enter_context(ZipFile(BytesIO(archive.read(inner))))
        metadata = archive.read(named_member(archive,"description.txt")).decode("cp1252")
        descriptions = {" ".join(line.lower().split()) for line in metadata.splitlines()}
        if not {"fs1 volume flow l/min 10 hz","ps2 pressure bar 100 hz"}.issubset(descriptions):
            raise ValueError("Unexpected FS1 or PS2 unit or sampling-rate metadata")
        labels = read_at(archive, "profile.txt", index, 5)
        fields = ("cooler","valve","pump","accumulator","stable_flag")
        if labels != [Decimal(entry[k]) for k in fields]:
            raise ValueError("Selected cycle and raw profile disagree")
        flow = read_at(archive,"FS1.txt",index,600)
        pressure = read_at(archive,"PS2.txt",index,6000)

    # Check all 12 original FS1 window means and the four exact boundary
    # samples against independently archived, previously reported records.
    panel = reference_row(root, "runs/u-pilot-02/panel.csv", "archive_row_0", index)
    edges = reference_row(root, "runs/u-paper-window-registration-diagnostic/edge_and_phase_predictions.csv",
                          "row_zero_based", index)
    for j in range(12):
        if sum(flow[50*j:50*(j+1)])*10**6 != Decimal(panel[f"q{j}_integer"]):
            raise ValueError(f"Unexpected FS1 readings at window {j}")
    for i in (0,50,500,550):
        if flow[i] != Decimal(edges[f"x{i}_l_min"]):
            raise ValueError(f"Boundary reading at {i/10} s differs from archived row")
    delta_first,delta_last = (flow[50]-flow[0])/50, (flow[550]-flow[500])/50
    if delta_first-delta_last != Decimal(edges["d_change_l_min"]):
        raise ValueError("The plotted edge samples do not reproduce the saved contrast displacement")
    output = {
        "selection":"first stable, grade-0 pilot row in archive order", "archive_row_zero_based":index,
        "cooler_percent":int(labels[0]), "valve_percent":int(labels[1]), "accumulator_bar":int(labels[3]),
        "grade":int(labels[2]), "fs1_samples_at_10_Hz":len(flow), "ps2_samples_at_100_Hz":len(pressure),
        "mean_flow_l_min":[float(sum(flow[a*10:b*10])/50) for a,b in WINDOWS],
        "mean_pressure_bar":[float(sum(pressure[a*100:b*100])/500) for a,b in WINDOWS],
        "fs1_at_0_5_50_55_s_l_min":[float(flow[i]) for i in (0,50,500,550)],
        "delayed_first_mean_change_l_min":float(delta_first), "delayed_late_mean_change_l_min":float(delta_last),
        "delayed_contrast_change_l_min":float(delta_first-delta_last)}
    return np.asarray(flow,dtype=float),np.asarray(pressure,dtype=float),output


def draw(flow, pressure, path):
    plt.rcParams.update({
        "font.family":"DejaVu Sans","font.size":8.0,"axes.labelsize":8.1,
        "axes.titlesize":8.8,"axes.titleweight":"semibold","xtick.labelsize":7.5,
        "ytick.labelsize":7.5,"text.color":".09","axes.labelcolor":".09","axes.edgecolor":".14",
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.65,
        "xtick.major.size":3,"ytick.major.size":3,"xtick.major.width":.65,"ytick.major.width":.65,
        "pdf.fonttype":42,"path.simplify":False,
    })
    fig = plt.figure(figsize=(7.03,3.03))
    grid = fig.add_gridspec(2,2,width_ratios=(1.7,1),left=.095,right=.977,
                            bottom=.16, top=.88, hspace=.44,wspace=.27)
    full_flow = fig.add_subplot(grid[0,0])
    full_pressure = fig.add_subplot(grid[1,0],sharex=full_flow)
    for axis,data,rate,label in [(full_flow,flow,10,"Flow (L/min)"),(full_pressure,pressure,100,"Pressure (bar)")]:
        for a,b in WINDOWS:
            axis.axvspan(a,b,color=".85",lw=0,zorder=-2)
        axis.plot(np.arange(len(data))/rate,data,color=".12",lw=.8)
        axis.set(xlim=(0,60),xticks=range(0,61,10),ylabel=label)
        axis.set_axisbelow(True)
        axis.grid(axis="y",color=".78",lw=.45,ls=(0,(1,3)))
        axis.tick_params(pad=2)
    full_flow.set_title("(a)  Main-circuit flow (FS1)",loc="left",pad=4)
    full_flow.set_ylim(-1.0,22.5)
    full_flow.set_yticks([0,10,20])
    full_flow.tick_params(labelbottom=False)
    full_pressure.set_title("(b)  Pressure (PS2)",loc="left",pad=4)
    full_pressure.set(ylim=(-7,176),yticks=[0,80,160],xlabel="Time in cycle (s)")

    # The gap on the right is a deliberately broken time axis: no lines are
    # drawn from 0.5 to 5.0 s. Only 5.0 s enters the shifted first window.
    detail = grid[:,1].subgridspec(1,2,width_ratios=(4.25,1.0),wspace=.095)
    early = fig.add_subplot(detail[0,0])
    endpoint = fig.add_subplot(detail[0,1],sharey=early)
    early.set_title("(c)  First-window boundary",loc="left",pad=4)
    early.plot(np.arange(6)/10,flow[:6],color=".12",lw=.9,marker="o",markersize=3.4,
               markerfacecolor="white",markeredgewidth=.85)
    early.plot(0,flow[0],"o",color=".12",ms=4,zorder=3)
    early.annotate(f"Leaves\n{flow[0]:.3f}",(0,flow[0]),(.091,7.55),
                   textcoords="data",va="center",ha="left",fontsize=7.4)
    early.set(xlim=(-.025,.515),ylim=(-.42,8.2),xticks=(0,.1,.2,.3,.4,.5),
              yticks=(0,2,4,6,8),ylabel="FS1 (L/min)")
    early.set_xticklabels(("0",".1",".2",".3",".4",".5"))
    early.tick_params(pad=2.4)
    endpoint.spines["left"].set_visible(False)
    endpoint.tick_params(left=False,labelleft=False,pad=2.4)
    endpoint.plot(5,flow[50],marker="s",ms=4.2,color=".12",zorder=3)
    endpoint.set(xlim=(4.96,5.04),xticks=[5.0],xticklabels=["5.0"])
    endpoint.annotate(f"Enters\n{flow[50]:.3f}",(5,flow[50]),(5,1.0),
                      ha="center",va="bottom",fontsize=7.4,
                      arrowprops={"arrowstyle":"-","lw":.65,"color":".34","shrinkB":3})
    for axis in (early,endpoint):
        axis.grid(axis="y",ls=(0,(1,3)),lw=.45,color=".74")
        axis.set_axisbelow(True)
    # Broken-axis marks have the same physical dimensions in both subaxes.
    for axis,x in [(early,1), (endpoint,0)]:
        axis.plot(x,0,marker=[(-.6,-1),(.6,1)],markersize=6,linestyle="none",
                  color=".17",mew=.85,transform=axis.transAxes,clip_on=False)
    left,right=early.get_position().x0,endpoint.get_position().x1
    fig.text((left+right)/2,.055,"Time in cycle (s)",ha="center",va="center",fontsize=8.1)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,metadata={"Title":"Observed ZeMA FS1 and PS2 cycle and its averaging boundaries",
                               "Creator":"Matplotlib","CreationDate":None})
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True,help="Public ZeMA data.zip or the UCI wrapper ZIP")
    parser.add_argument("--package-root",type=Path,default=ROOT,help="Directory containing cycle_split.csv and runs/")
    parser.add_argument("--out",type=Path,default=ROOT/"figures/observed-cycle.pdf",help="Output vector PDF")
    args=parser.parse_args()
    fs1,ps2,summary=load_selected(args.archive,args.package_root)
    draw(fs1,ps2,args.out)
    print(json.dumps(dict(**summary,pdf_sha256=sha256(args.out.read_bytes()).hexdigest()),indent=2))
