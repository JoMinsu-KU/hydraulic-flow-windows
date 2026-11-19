"""Check appendix tables and physical interpretation against the released records."""
import csv
from collections import Counter, defaultdict
from decimal import Decimal as D
from fractions import Fraction
import json
from math import isclose
from pathlib import Path
from statistics import median

import replay_primary as primary


ROOT = Path(__file__).resolve().parent


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def csv_rows(relative):
    path = ROOT / relative
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def grade_table(centers):
    held = defaultdict(list)
    for row in primary.published_panel():
        held[row[1]].append(row)
    table = []
    for setting, rows in sorted(held.items()):
        one = max(primary.optimum(primary.cell(rows, j), 30_000_000) for j in range(12))
        def pred(s):
            return min(range(3), key=lambda c: abs(2*(s[0]-s[10])-centers[c]))
        two = sum(pred(s) == grade for _,_,grade,s in rows)
        table.append([*setting,one,two,two-one])
    assert [r[-3:-1] for r in table] == [[19,13],[18,23],[10,30],[10,30],[10,30],
                                         [10,25],[10,23],[10,28],[10,29],[10,30]]
    methods = ("fixed","nested","one_wave","two_wave")
    cells = defaultdict(Counter)
    observations = defaultdict(set)
    for r in csv_rows("runs/u-paper-groups-01/predictions.csv"):
        mode, group, row, grade = r["mode"], int(r["group_id"]), int(r["row"]), int(r["grade"])
        assert row not in observations[mode]
        observations[mode].add(row)
        for method in methods:
            cells[(mode,group,method,grade)][int(r[method])] += 1
    outcomes = load("runs/u-paper-groups-01/results.json")["results"]
    grade_recall = {}
    for mode, summary in outcomes.items():
        groups = sorted({g for m,g,_,_ in cells if m == mode})
        assert len(observations[mode]) == summary["n_rows"] and len(groups) == summary["n_groups"]
        grade_recall[mode] = {}
        for method in methods:
            archive = summary["methods"][method]
            grade_recall[mode][method] = []
            for c in range(3):
                counters = [cells[(mode,g,method,c)] for g in groups]
                counts = sum(counters,Counter())
                assert [counts[k] for k in range(3)] == archive["confusion_raw"][c]
                recall = sum(Fraction(t[c], t.total()) for t in counters) / len(groups)
                assert isclose(float(recall), archive["confusion_row_proportions_group_mean"][c][c])
                grade_recall[mode][method].append(round(float(recall)*100,2))
    assert outcomes["strictly_past"]["methods"]["nested"]["confusion_raw"][2] == [0,169,131]
    return {"held_counts":table,"balanced_grade_recall_percent":grade_recall}


def margin_table(centers):
    mu = [D(k)/(2*primary.SCALE) for k in centers]
    cut = [(x+y)/2 for x,y in zip(mu,mu[1:])]
    assert cut == [D("-7.525365"),D("-7.350700")]
    def pred(v):
        return min(range(3),key=lambda c:abs(v-mu[c]))
    count = 0
    for r in csv_rows("runs/u-followup-b/cycle_robust.csv"):
        x, a, e = (D(r[k]) for k in ("d_l_min","gain_percent","window_error_l_min"))
        c = int(r["grade"])
        radius = abs(x)*a/100 + 2*e
        lo,hi = x-radius, x+radius
        signed = min(([x-cut[c-1]] if c>0 else [])+([cut[c]-x] if c<2 else []))
        assert abs(signed-D(r["signed_margin_l_min"])) < D("1e-12")
        assert abs(lo-D(r["lo_l_min"])) < D("1e-12") and abs(hi-D(r["hi_l_min"])) < D("1e-12")
        assert pred(x)==int(r["prediction"]) and (pred(lo)==pred(hi)==c)==bool(int(r["robust"]))
        count += 1
    assert count==6000
    endpoints = defaultdict(dict)
    for r in csv_rows("runs/u-wave-gain-axis-audit/refined-frontier-endpoints.csv"):
        k=(r["axis"],float(r["fixed_gain_fraction"]),r["pass_or_fail"],float(r["radius"]))
        rid=int(r["row"])
        assert rid not in endpoints[k] and r["independent_status"] in ("1","-1")
        endpoints[k][rid]=int(r["independent_status"])
    frontier = load("runs/u-wave-gain-axis-audit/frontier-refined.json")["frontier"]
    assert len(endpoints)==2*len(frontier)==20
    for v in frontier:
        for side, expected in [("pass",255),("fail",254)]:
            ids=endpoints[v["axis"],v["fixed_gain_fraction"] or 0.,side,v[side+"_radius"]]
            assert len(ids)==300 and sum(s==1 for s in ids.values())==v[side+"_lower_correct"]==v[side+"_upper_correct"]==expected
    comparisons = Counter((int(r["official"]),int(r["independent"])) for r in
                          csv_rows("runs/u-wave-gain-axis-audit/region-row-agreement.csv"))
    assert comparisons.total()==54000 and comparisons[(1,-1)]==comparisons[(-1,1)]==0
    return dict(two_median_boundaries=list(map(float,cut)),pilot_intervals_checked=count,
                frontier_brackets=len(frontier),endpoints_checked=sum(map(len,endpoints.values())),
                two_dimensional_comparisons=comparisons.total())


def physical_summary():
    phase = load("runs/u-paper-applicability/result.json")["phase"]["FS1_L_min"]
    series = defaultdict(list)
    for r in csv_rows("runs/u-pilot-02/panel.csv"):
        for j,key in [(0,"first"),(10,"late")]:
            series[(r["grade"],key)].append(int(r[f"q{j}_integer"])/primary.SCALE)
    def quantile(arr, prob):
        # Linear interpolation between successive order statistics.
        vals=sorted(arr)
        pos=(len(vals)-1)*prob
        i=int(pos)
        return vals[i]+(pos-i)*(vals[min(i+1,len(vals)-1)]-vals[i])
    for (grade,window), vals in series.items():
        assert len(vals)==100
        for k,p in [("p05",.05),("median",.5),("p95",.95)]:
            assert isclose(quantile(vals,p), phase[grade][window][k])
    diag=load("runs/u-paper-window-registration-diagnostic/results.json")
    timing=defaultdict(list)
    for r in csv_rows("runs/u-paper-window-registration-diagnostic/edge_and_phase_predictions.csv"):
        e=(D(r["x50_l_min"])-D(r["x0_l_min"]))/50
        l=(D(r["x550_l_min"])-D(r["x500_l_min"]))/50
        ds=e-l
        assert ds < 0 and ds==D(r["d_change_l_min"]) and ds==D(r["d_shifted_l_min"])-D(r["d_nominal_l_min"])
        assert e==D(r["first_mean_change_l_min"]) and l==D(r["last_mean_change_l_min"])
        timing[int(r["grade"])].append((e,l,ds))
    assert len(timing)==3
    for ref in diag["by_grade"]:
        vals = timing[ref["grade"]]
        for j,k in enumerate(("median_first_mean_change_l_min","median_last_mean_change_l_min","median_change_in_delta_l_min")):
            assert isclose(float(median(t[j] for t in vals)),ref[k])
    return {"flow_quantiles_checked":3*len(series),"shifted_contrasts_replayed":sum(map(len,timing.values()))}


def external_dates():
    cells=defaultdict(Counter)
    for r in csv_rows("runs/u-ponar-public-transfer/predictions.csv"):
        k=(r["analysis"],r["method"],r["role"],r["plate"],r["recording_day"])
        cells[k].update(n=1,correct=int(r["truth"]==r["prediction"]),robust=int(r["robust_correct"]))
    summary=load("runs/u-ponar-public-transfer/results.json")
    check_days=0
    for analysis,exp in summary["experiments"].items():
        for method,info in exp["methods"].items():
            for r in info["by_day"]:
                day=cells[(analysis,method,r["role"],r["plate"],r["day"])]
                assert (day["n"],day["correct"],day["robust"]) == (r["n"],r["correct"],r["robust_correct"])
                check_days+=1
    assert len(cells)==check_days
    entries=sum(day["n"] for day in cells.values())
    assert entries==9620
    return dict(prediction_records_checked=entries,method_by_day_cells=check_days)


if __name__ == "__main__":
    fixed_centers=load("protocols/confirmation-lock.json")["double_centers_integer"]
    results={"appendix_B":grade_table(fixed_centers),"appendix_C":margin_table(fixed_centers),
             "phase_evidence":physical_summary(),"appendix_D":external_dates()}
    print(json.dumps(results,indent=2))
