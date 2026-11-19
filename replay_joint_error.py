"""Refit and certify the reported 54.9-s waveform models at their joint error point.

    python replay_joint_error.py --archive /path/to/data.zip

For each pilot cycle, the centered classifier must be correct for every
relative gain in ±0.25% and independent constant residual in ±0.005 L/min
per acquired window. In a pair, only the difference between residuals
survives centering. The parameter h is half that difference. Each feasible h
can be realized by residuals +h and -h. In a single window, h is always zero.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numba
import numpy as np
import sklearn

from methods.forest_bounds import analyze, flatten, raw_transform
from replay_waveforms import LAYOUTS, ROOT, centered_features, fit_forest, read_waveforms


GAIN = 0.0025
WINDOW_ERROR = 0.005
MAX_BOXES = 8191
REFERENCE = Path("runs/u-wave-gain-axis-audit/region-row-agreement.csv")


def published_statuses(path):
    expected = {"single_exact": {}, "pair_exact": {}}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for r in csv.DictReader(stream):
            if int(r["gain_ppm"]) != 2500:
                continue
            # A common single-window residual cancels. The archived comparison
            # uses the equivalent radius E=0 for this layout.
            target = {"single": 0, "pair": 5000}.get(r["layout"])
            if target is None or int(r["error_micro"]) != target:
                continue
            batch = expected[r["layout"] + "_exact"]
            row, grade = int(r["row"]), int(r["grade"])
            status = int(r["independent"])
            if row in batch or status != int(r["official"]) or status not in (-1, 1):
                raise ValueError("The released row-level decisions disagree or contain a duplicate")
            batch[row] = grade, status
    if any(len(batch) != 300 for batch in expected.values()):
        raise ValueError("Incomplete published reference at the joint error point")
    return expected


def certify_model(model, features, grades, split):
    expected_shape = (len(grades), 62)
    if features.shape != expected_shape or not np.isfinite(features).all():
        raise ValueError(f"Expected finite features of shape {expected_shape}")
    if (len(model.estimators_) != 96 or not np.array_equal(model.classes_, [0, 1, 2])
            or max(t.tree_.max_depth for t in model.estimators_) >= 127):
        raise ValueError("Forest incompatible with the numerical and traversal bounds of this replay")
    trees = flatten(model)
    signs = np.repeat([1., -1.], 25) if split else np.zeros(50)
    max_h = WINDOW_ERROR if split else 0.
    rows = np.array([analyze(f, signs, int(y), GAIN, max_h, MAX_BOXES, *trees)
                     for f, y in zip(features, grades)])
    status = rows[:, 0].astype(int)
    failed = status == -1
    wrong = rows[failed]
    if len(wrong):
        gain, h = wrong[:, 1], wrong[:, 2]
        if np.any(np.abs(gain) > GAIN) or np.any(np.abs(h) > max_h):
            raise ValueError("An error witness is outside the specified box")
        replay = raw_transform(features[failed, :50], signs, gain, h)
        predictions = model.predict(replay)
        if np.any(predictions == grades[failed]) or np.any(predictions != wrong[:, 3]):
            raise ValueError("A proposed error witness failed in the fitted scikit-learn model")
    return status


def replay(archive, root):
    data, ids, grades, fit = read_waveforms(archive, root/"cycle_split.csv")
    truth = grades[~fit]
    pilot_ids = ids[~fit]
    expected = published_statuses(root/REFERENCE)
    results = {}
    for layout, indices in LAYOUTS.items():
        split = layout == "pair_exact"
        features = centered_features(data[:, indices])
        model = fit_forest(features[fit], grades[fit])
        nominal = int(np.count_nonzero(model.predict(features[~fit]) == truth))
        status = certify_model(model, features[~fit], truth, split)
        verified = {int(i): (int(y), int(s)) for i, y, s in zip(pilot_ids, truth, status)}
        if verified != expected[layout]:
            rows = sorted(k for k in verified.keys() | expected[layout].keys()
                          if verified.get(k) != expected[layout].get(k))
            raise ValueError(f"{layout}: {len(rows)} decisions differ from the release, first row {rows[0]}")
        counts = Counter(status.tolist())
        if (nominal, counts[1], counts[0]) != ((297, 256, 0) if split else (227, 226, 0)):
            raise ValueError(f"{layout}: joint-error results differ from the reported counts")
        results[layout] = {
            "pilot_cycles": len(truth), "nominal_correct": nominal, "certified_correct": counts[1],
            "wrong_with_witness": counts[-1], "unresolved": counts[0],
            "balanced_guaranteed_accuracy": counts[1]/len(truth),
            "correct_by_grade_none_weak_severe": [int(np.count_nonzero((status == 1) & (truth == c)))
                                                  for c in (0, 1, 2)],
            "matched_published_status_and_grade": len(verified),
        }
    return {
        "fit_cycles": int(fit.sum()), "pilot_cycles": len(truth), "confirmation_signals_parsed": 0,
        "samples_per_layout": 50, "last_sample_s": 54.9, "gain_half_width_fraction": GAIN,
        "window_residual_half_width_l_min": WINDOW_ERROR, "timing_shift_s": 0,
        "low_flow_clipping": False, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "numba": numba.__version__, "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="ZeMA data.zip or the UCI wrapper ZIP")
    parser.add_argument("--package-root", type=Path, default=ROOT,
                        help="directory containing cycle_split.csv and runs/ (defaults to script location)")
    args = parser.parse_args()
    print(json.dumps(replay(args.archive, args.package_root), indent=2))
