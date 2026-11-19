"""Select an interior reference window on training conditions and test clock shifts on the reused pilot."""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from replay_primary import optimum
from replay_waveforms import ROOT, SCALE, read_waveforms


WIDTH = 50
LATE = 500
SHIFTS = (-1, 0, 1)
EARLY_STARTS = tuple(range(5, 46, 5))


def window_sum(flow, start):
    return flow[:, start:start + WIDTH].sum(axis=1)


def differences(flow, start, shifts):
    return np.column_stack([window_sum(flow, start + delta) - window_sum(flow, LATE + delta)
                            for delta in shifts])


def centers_twice(nominal, grades):
    # Twice the median preserves half-integer ties for even grade counts.
    centers = []
    for grade in range(3):
        values = np.sort(nominal[grades == grade])
        n = len(values)
        centers.append(int(values[(n-1)//2] + values[n//2]))
    return np.asarray(centers)


def predict(contrasts, centers):
    return np.abs(2 * contrasts[:, :, None] - centers).argmin(axis=2)


def fold_score(correct, grades):
    return sum(Fraction(int(correct[grades == c].sum()), int((grades == c).sum()))
               for c in range(3)) / 3


def select_early(flow, grades, groups):
    scores = {}
    for start in EARLY_STARTS:
        diff = differences(flow, start, SHIFTS)
        total = Fraction(0)
        for group in np.unique(groups):
            held = groups == group
            centers = centers_twice(diff[~held, 1], grades[~held])
            hits = (predict(diff[held], centers) == grades[held, None]).all(axis=1)
            total += fold_score(hits, grades[held])
        scores[start] = total / len(np.unique(groups))
    return max(EARLY_STARTS, key=scores.get), scores


def single_ceiling(flow, grades, groups):
    starts = sorted(set(range(0, 600, WIDTH)) | set(EARLY_STARTS))
    sums = np.column_stack([window_sum(flow, j) for j in starts])
    count = 0
    for group in np.unique(groups):
        ix = groups == group
        assert np.bincount(grades[ix]).tolist() == [10, 10, 10]
        count += max(optimum(sorted(zip(sums[ix, j].tolist(), grades[ix].tolist())), 30_000_000)
                     for j in range(len(starts)))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    split_path = ROOT / "cycle_split.csv"
    flow, ids, grades, fit = read_waveforms(args.archive, split_path)
    with split_path.open(encoding="utf-8-sig", newline="") as f:
        settings = [tuple(int(row[k]) for k in ("cooler", "valve", "accumulator"))
                    for row in csv.DictReader(f)]
    _, groups = np.unique([settings[i] for i in ids], axis=0, return_inverse=True)
    start, scores = select_early(flow[fit], grades[fit], groups[fit])
    diff = differences(flow, start, SHIFTS)
    centers = centers_twice(diff[fit, 1], grades[fit])
    pred = predict(diff[~fit], centers)
    truth = grades[~fit]
    correct = pred == truth[:, None]
    baseline = differences(flow, 0, (0, 1))
    basepred = predict(baseline[~fit], centers_twice(baseline[fit, 0], grades[fit]))
    basehits = basepred == truth[:, None]
    assert basehits.sum(axis=0).tolist() == [271, 129] and basehits.all(axis=1).sum() == 107
    robust = correct.all(axis=1)
    summary = {
        "analysis": "post-hoc timing sensitivity, reused pilot, no confirmation cycles",
        "fit_n": int(fit.sum()), "pilot_n": len(truth),
        "candidate_starts_samples": EARLY_STARTS, "nominal_late_start_sample": LATE,
        "window_length_samples": WIDTH, "common_clock_shifts_samples": SHIFTS,
        "training_logo_balanced_joint_accuracy": {str(j): str(s) for j, s in scores.items()},
        "selection_tie_break": "earliest start", "selected_start_sample": start,
        "training_medians_l_min": (centers / (2 * WIDTH * SCALE)).tolist(),
        "correct_early_nominal_late": correct.sum(axis=0).tolist(),
        "correct_at_all_three": int(robust.sum()),
        "all_three_correct_by_grade": [int(robust[truth == c].sum()) for c in range(3)],
        "original_nominal_plus1_and_both": [271, 129, 107],
        "pilot_single_window_aligned_oracle_at_B_0p30": single_ceiling(flow[~fit], truth, groups[~fit]),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "results.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    with (args.out / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cycle_zero_based", "grade", "pred_minus1", "pred_nominal", "pred_plus1",
                         "correct_at_all_three"])
        for i, y, p, h in zip(ids[~fit], truth, pred, robust):
            writer.writerow([i, y, *p, int(h)])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
