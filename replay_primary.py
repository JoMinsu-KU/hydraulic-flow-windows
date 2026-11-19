"""Replay the pre-specified ZeMA panel from archived means or from the public ZIP."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from decimal import Decimal
from fractions import Fraction
from itertools import combinations
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
SCALE = 50_000_000  # 50 samples per mean, raw units of 1e-6 L/min
TARGET = Fraction(17, 20)
CENTER_FILE = HERE / "protocols" / "confirmation-lock.json"


def by_name(archive, name):
    matches = [member for member in archive.namelist() if PurePosixPath(member).name.lower() == name]
    if len(matches) != 1:
        raise ValueError("Expected one " + name + " in the archive")
    return matches[0]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def published_panel():
    rows = read_csv(HERE / "runs/u-confirm-01/panel.csv")
    return [(int(row["archive_row_0"]), tuple(int(row[field]) for field in ("cooler","valve","accumulator")),
             int(row["grade"]), tuple(int(row[f"q{j}_integer"]) for j in range(12))) for row in rows]


def raw_panel(zip_path, expected_panel, frozen_centers):
    splits = read_csv(HERE / "cycle_split.csv")
    training = defaultdict(list)
    output = []
    with ExitStack() as stack:
        archive = stack.enter_context(ZipFile(zip_path))
        if not any(PurePosixPath(n).name.lower() == "fs1.txt" for n in archive.namelist()):
            archive = stack.enter_context(ZipFile(BytesIO(archive.read(by_name(archive,"data.zip")))))
        with archive.open(by_name(archive,"profile.txt")) as profile, archive.open(by_name(archive,"fs1.txt")) as fs:
            for i,(row,line,labels) in enumerate(zip(splits,fs,profile,strict=True)):
                fields = tuple(int(Decimal(token.decode("ascii"))) for token in labels.split())
                expected = tuple(int(row[k]) for k in ("cooler","valve","pump","accumulator","stable_flag"))
                assert i == int(row["cycle_zero_based"]) and fields == expected
                if row["role"] not in {"fit","confirm"}:
                    continue
                samples_decimal = [Decimal(token.decode("ascii")) * 1_000_000 for token in line.split()]
                assert len(samples_decimal) == 600 and all(v == int(v) for v in samples_decimal)
                values = list(map(int,samples_decimal))
                sums = tuple(sum(values[j:j+50]) for j in range(0,600,50))
                g = (fields[0],fields[1],fields[3])
                if row["role"] == "fit":
                    training[fields[2]].append(sums[0]-sums[10])
                else:
                    output.append((i,g,fields[2],sums))
    fitted = []
    for c in range(3):
        ordered = sorted(training[c])
        n = len(ordered)
        fitted.append(ordered[(n-1)//2] + ordered[n//2])
    assert fitted == frozen_centers and output == expected_panel
    return output


def cell(rows,j):
    return sorted((means[j],grade) for _,_,grade,means in rows)


def optimum(ordered,diameter):
    # The target panel has 10 cycles per grade per condition.
    # Each selected row carries the same weight.
    x = [value for value,_ in ordered]
    prefix = [0]
    best_for_grade = [0,0,0]
    for i,(value,grade) in enumerate(ordered):
        far = bisect_left(x,value-diameter,0,i)
        ending = 1 + max(prefix[far],best_for_grade[grade])
        best_for_grade[grade] = max(best_for_grade[grade],ending)
        prefix.append(max(prefix[-1],ending))
    return prefix[-1]


def verify(rows,centers):
    groups = defaultdict(list)
    for row in rows:
        groups[row[1]].append(row)
    assert len(groups)==10 and all(Counter(r[2] for r in g)=={0:10,1:10,2:10} for g in groups.values())
    ordered = [[cell(g,j) for j in range(12)] for g in groups.values()]
    def upper(diameter):
        return sum(max(optimum(item,diameter) for item in group) for group in ordered)
    def predict(means):
        double = 2*(means[0]-means[10])
        return min(range(3),key=lambda c:abs(double-centers[c]))
    correct = sum(predict(r[3])==r[2] for r in rows)
    scenario = upper(30_000_000)   # diameter corresponding to ±0.30 L/min
    events = {0}
    for group in ordered:
        for item in group:
            events.update(b-a for (a,y),(b,z) in combinations(item,2) if y!=z)
    events = sorted(events)
    left,right = 0,len(events)
    while left < right:
        mid = (left+right)//2
        if Fraction(upper(events[mid]),len(rows)) < TARGET:
            right = mid
        else:
            left = mid+1
    assert 0 < left < len(events)
    before,first = events[left-1:left+1]
    assert (correct,scenario,upper(before),upper(first))==(261,117,255,254)
    assert Fraction(first,2*SCALE)==Fraction(5932,100000)
    return {"cycles":len(rows),"two_mean_correct":correct,"one_mean_ceiling_at_B_0.30":scenario,
            "previous_event_l_min":float(Fraction(before,2*SCALE)),"previous_ceiling_count":upper(before),
            "critical_half_width_l_min":float(Fraction(first,2*SCALE)),"critical_ceiling_count":upper(first)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Optional original ZeMA data.zip or UCI wrapper ZIP.")
    options=parser.parse_args()
    centers=json.loads(CENTER_FILE.read_text(encoding="utf-8"))["double_centers_integer"]
    rows=published_panel()
    if options.archive:
        rows=raw_panel(options.archive,rows,centers)
    print(json.dumps(verify(rows,centers),indent=2))


if __name__ == "__main__":
    main()
