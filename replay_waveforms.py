"""Refit the two phase-aligned, centered FS1 waveform models reported at 54.9 s.

Run with Python 3.10+, NumPy and scikit-learn:
    python replay_waveforms.py --archive /path/to/data.zip
The release contains the split and row-level predictions for comparison.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from decimal import Decimal
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier


ROOT = Path(__file__).resolve().parent
SCALE = 1_000_000
PROFILE_FIELDS = ("cooler", "valve", "pump", "accumulator", "stable_flag")
LAYOUTS = {
    "single_exact": np.arange(500, 550),
    "pair_exact": np.r_[np.arange(1, 50, 2), np.arange(501, 550, 2)],
}


def member(archive, basename):
    matches = [info for info in archive.infolist()
               if PurePosixPath(info.filename).name.lower() == basename]
    if len(matches) != 1:
        raise ValueError(f"Expected one {basename} member, found {len(matches)}")
    return matches[0]


def read_waveforms(archive_path, split_path):
    with split_path.open(encoding="utf-8-sig", newline="") as stream:
        splits = list(csv.DictReader(stream))
    if len(splits) != 2205:
        raise ValueError("Expected all 2,205 original cycle assignments")

    signals, rows, labels, roles, groups = [], [], [], [], []
    with ExitStack() as stack:
        archive = stack.enter_context(ZipFile(archive_path))
        if not any(PurePosixPath(name).name.lower() == "fs1.txt" for name in archive.namelist()):
            nested = member(archive, "data.zip")
            if nested.file_size > 128 * 1024**2:
                raise ValueError("Nested data.zip exceeds the size of the published archive")
            archive = stack.enter_context(ZipFile(BytesIO(archive.read(nested))))
        flow = stack.enter_context(archive.open(member(archive, "fs1.txt")))
        profile = stack.enter_context(archive.open(member(archive, "profile.txt")))
        for index, (split, line, conditions) in enumerate(zip(splits, flow, profile, strict=True)):
            if index != int(split["cycle_zero_based"]):
                raise ValueError(f"Split order differs at archive row {index}")
            if split["role"] not in ("fit", "pilot"):
                continue
            settings = tuple(int(split[key]) for key in PROFILE_FIELDS)
            if tuple(Decimal(word.decode("ascii")) for word in conditions.split()) != settings:
                raise ValueError(f"Archive profile and split disagree at row {index}")
            if settings[4] != 0:
                raise ValueError(f"Unstable cycle in the fit or pilot set at row {index}")
            values = np.fromstring(line, sep=" ")
            if values.size != 600 or not np.isfinite(values).all():
                raise ValueError(f"Invalid 60-s FS1 waveform at row {index}")
            integer = np.rint(values * SCALE).astype(np.int64)
            if np.any(np.abs(values - integer / SCALE) > 1e-13):
                raise ValueError(f"Unexpected decimal precision at row {index}")
            signals.append(integer)
            rows.append(index)
            labels.append(settings[2])
            roles.append(split["role"])
            groups.append((settings[0], settings[1], settings[3]))

    fit_groups = {g for g, role in zip(groups, roles) if role == "fit"}
    pilot_cells = Counter((g, y) for g, y, role in zip(groups, labels, roles) if role == "pilot")
    pilot_groups = {g for g, _ in pilot_cells}
    if (Counter(roles) != {"fit": 849, "pilot": 300} or len(fit_groups) != 28
            or len(pilot_groups) != 10 or fit_groups & pilot_groups
            or any(pilot_cells[g, label] != 10 for g in pilot_groups for label in (0, 1, 2))):
        raise ValueError("Unexpected group separation or grade counts")
    return np.stack(signals), np.array(rows), np.array(labels), np.asarray(roles) == "fit"


def centered_features(samples):
    # Integer centering before unit conversion reproduces the training implementation.
    centered = (50 * samples - samples.sum(axis=1, keepdims=True)) / (50 * SCALE)
    return np.column_stack((centered, centered.std(axis=1), np.ptp(centered, axis=1),
                            centered.reshape(-1, 10, 5).mean(axis=2)))


def fit_forest(features, labels):
    model = ExtraTreesClassifier(n_estimators=96, min_samples_leaf=2, max_features=1.0,
                                 class_weight="balanced", random_state=181119, n_jobs=1)
    return model.fit(features, labels)


def archived_predictions(path):
    predictions = {key: {} for key in LAYOUTS}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for record in csv.DictReader(stream):
            layout = record["layout"]
            if (record["deadline_window"] != "10" or layout not in predictions
                    or record["method"] != "centered_wave"):
                continue
            row = int(record["row_zero_based"])
            if row in predictions[layout]:
                raise ValueError(f"Duplicate reference prediction at row {row} for {layout}")
            predictions[layout][row] = (int(record["grade"]), int(record["b_centi_0"]))
    return predictions


def replay(archive_path, root):
    data, row_ids, y, fit = read_waveforms(archive_path, root / "cycle_split.csv")
    expected = archived_predictions(root / "runs/u-followup-c/predictions.csv")
    pilot_rows = row_ids[~fit]
    results = {}
    for layout, indices in LAYOUTS.items():
        features = centered_features(data[:, indices])
        model = fit_forest(features[fit], y[fit])
        predictions = model.predict(features[~fit])
        actual = {int(row): (int(label), int(pred)) for row, label, pred
                  in zip(pilot_rows, y[~fit], predictions)}
        if actual != expected[layout]:
            mismatches = sorted(row for row in actual.keys() | expected[layout].keys()
                                if actual.get(row) != expected[layout].get(row))
            raise ValueError(f"{layout}: {len(mismatches)} archived grade/prediction mismatches, "
                             f"first archive row {mismatches[0]}")
        correct = int(np.count_nonzero(predictions == y[~fit]))
        if correct != {"single_exact": 227, "pair_exact": 297}[layout]:
            raise ValueError(f"{layout}: unexpected correct count {correct}")
        results[layout] = {"correct": correct, "cycles": len(actual),
                           "balanced_accuracy": correct / len(actual),
                           "matched_archived_predictions": len(actual)}
    return {"fit_cycles": int(fit.sum()), "pilot_cycles": len(pilot_rows),
            "raw_samples_per_cycle_and_layout": 50, "last_sample_s": 54.9,
            "numpy": np.__version__, "scikit_learn": sklearn.__version__, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="ZeMA data.zip or the UCI wrapper ZIP")
    parser.add_argument("--package-root", type=Path, default=ROOT,
                        help="directory containing cycle_split.csv and runs/ (defaults to script location)")
    args = parser.parse_args()
    print(json.dumps(replay(args.archive, args.package_root), indent=2))
