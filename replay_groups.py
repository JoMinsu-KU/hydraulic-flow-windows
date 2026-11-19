"""Rebuild the 38 condition-held-out and 30 strictly-past fits from the ZeMA ZIP."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from fractions import Fraction
from io import BytesIO
from itertools import combinations
import json
from math import lcm
from pathlib import Path, PurePosixPath
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold

from replay_waveforms import LAYOUTS, ROOT, centered_features, fit_forest, member, read_waveforms


PAIRS = list(combinations(range(12), 2))
METHODS = ("nested", "fixed", "pressure", "one_wave", "two_wave")
MODES = ("condition_logo", "strictly_past")
RADII_MILLI = (0, 100, 300)
CONDITION_FIELDS = ("cooler", "valve", "accumulator")
SCALE = 50_000_000  # sum of 50 readings in millionths of a liter per minute


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def equal(value, expected, label):
    if value != expected:
        raise ValueError("Verification mismatch: " + label)


def close(value, expected, label):
    np.testing.assert_allclose(value, expected, rtol=0, atol=1e-12, err_msg=label)


def pressure_phases(archive_path, ids):
    wanted = set(ids)
    samples = []
    with ExitStack() as stack:
        archive = stack.enter_context(ZipFile(archive_path))
        if not any(PurePosixPath(n).name.lower() == "ps1.txt" for n in archive.namelist()):
            nested = member(archive, "data.zip")
            if nested.file_size > 128 * 1024**2:
                raise ValueError("Unexpected nested archive size")
            archive = stack.enter_context(ZipFile(BytesIO(archive.read(nested))))
        with archive.open(member(archive, "ps1.txt")) as source:
            for row, line in enumerate(source):
                if row not in wanted:
                    continue
                trace = np.fromstring(line, sep=" ")
                if trace.size != 6000 or not np.isfinite(trace).all():
                    raise ValueError(f"Invalid PS1 waveform at row {row}")
                samples.append(trace.reshape(12, 500).mean(axis=1))
    equal(len(samples), len(ids), "selected pressure rows")
    return np.stack(samples)


def doubled_medians(contrast, grades):
    centers = []
    for c in range(3):
        ordered = np.sort(contrast[grades == c], axis=0)
        centers.append(ordered[(len(ordered)-1)//2] + ordered[len(ordered)//2])
    return np.asarray(centers, dtype=np.int64)


def predict_contrasts(values, centers):
    return np.abs(2 * values[:, None, :] - centers[None, :, :]).argmin(axis=1)


def choose_pairs(contrast, pressure, grades, groups):
    # Recreate every inner validation fold. No outer prediction or recorded score
    # enters this ranking. Unequal grade counts need integer weights to preserve ties.
    held_groups = []
    votes = np.empty_like(contrast, dtype=np.int8)
    for fit, val in GroupKFold(n_splits=4).split(contrast, grades, groups):
        held_groups.append(np.unique(groups[val]).tolist())
        votes[val] = predict_contrasts(contrast[val], doubled_medians(contrast[fit], grades[fit]))
    _, cell_id, counts = np.unique(np.column_stack((groups, grades)), axis=0,
                                   return_inverse=True, return_counts=True)
    unit = lcm(*map(int, counts))
    weights = unit // counts[cell_id.ravel()]
    totals = ((votes == grades[:, None]) * weights[:, None]).sum(axis=0)
    pressure_mean = pressure.mean(axis=0)
    best = max(range(len(PAIRS)), key=lambda i: (totals[i],
               abs(pressure_mean[PAIRS[i][0]] - pressure_mean[PAIRS[i][1]]), -i))
    pressure_pair = tuple(sorted((int(pressure_mean.argmin()), int(pressure_mean.argmax()))))
    choices = [best, PAIRS.index((0, 10)), PAIRS.index(pressure_pair)]
    return choices, held_groups, (totals / (len(counts) * unit)).tolist()


def window_ceiling(means, grades, diameter):
    """Quadratic exact max-weight compatible set of sorted equal-width intervals."""
    order = np.argsort(means, kind="stable")
    values, labels = means[order].tolist(), grades[order].tolist()
    counts = Counter(labels)
    unit = lcm(*counts.values())
    best = []
    for i, (value, label) in enumerate(zip(values, labels)):
        previous = max((best[k] for k in range(i) if labels[k] == label
                        or value - values[k] > diameter), default=0)
        best.append(previous + unit // counts[label])
    return Fraction(max(best), 3 * unit)


def compute_fold(mode, target, train_groups, ids, groups, grades, contrast, pressure, waves):
    train = np.isin(groups, train_groups)
    test = groups == target
    choices, inner, scores = choose_pairs(contrast[train], pressure[train], grades[train], groups[train])
    medians = doubled_medians(contrast[train][:, choices], grades[train])
    columns = [predict_contrasts(contrast[test][:, choices], medians)]
    for x in waves:
        columns.append(fit_forest(x[train], grades[train]).predict(x[test])[:, None])
    log = {"mode": mode, "target_id": target, "train_ids": train_groups,
           "inner_heldout_group_ids": inner, "pairs": [list(PAIRS[j]) for j in choices],
           "train_n": int(train.sum()), "test_n": int(test.sum()),
           "train_rows": [int(ids[train].min()), int(ids[train].max())],
           "test_rows": [int(ids[test].min()), int(ids[test].max())],
           "doubled_centers_integer_by_grade": medians.tolist(), "cv_scores_by_lexicographic_pair": scores}
    if mode == "strictly_past" and log["train_rows"][1] >= log["test_rows"][0]:
        raise ValueError("A future or overlapping row entered a forward fit")
    return log, np.column_stack(columns)


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def index_rows(rows, keys):
    indexed = {tuple(row[k] for k in keys): row for row in rows}
    equal(len(indexed), len(rows), f"unique rows for {keys}")
    return indexed


def manuscript_table(root, development):
    # Keep the ten-condition primary endpoint isolated. Only its published
    # counts are ingested here; replay_primary.py independently refits it.
    holdout = read_json(root / "runs/u-confirm-01/raw-reconstruction-audit.json")["B_0p30"]
    groups = holdout["per_group"]
    total = sum(row["n"] for row in groups)
    confirmed = {}
    for key, field in (("U1_B030", "oracle_single_BA"), ("fixed", "two_mean_BA")):
        correct = sum(round(row["n"] * row[field]) for row in groups)
        confirmed[key] = correct / total
        close(confirmed[key], holdout["single_mean_oracle_BA" if key == "U1_B030" else "two_mean_BA"],
              "held-out archived endpoint")
    equal((len(groups), total), (10, 300), "held-out archived panel")
    rows = []
    for method in ("U1_B030", "fixed", "nested", "one_wave", "two_wave"):
        entry = {"method": method, "held_out_10_percent": ""}
        if method in confirmed:
            entry["held_out_10_percent"] = f"{100 * confirmed[method]:.2f}"
        for row in development:
            entry[row["mode"] + "_percent"] = f"{100 * row[method]:.2f}"
        rows.append(entry)
    return rows


def replay(archive, root, out):
    if out.exists():
        raise FileExistsError(f"Choose a new --out directory: {out}")
    clock = perf_counter()
    split = root / "cycle_split.csv"
    raw, ids, grades, fit = read_waveforms(archive, split)
    profile = read_csv(split)
    keys, groups = np.unique([[int(profile[row][k]) for k in CONDITION_FIELDS] for row in ids],
                             axis=0, return_inverse=True)
    groups = groups.ravel()
    equal(len(keys), 38, "number of stable development conditions")
    pressure = pressure_phases(archive, ids)
    qsums = raw.reshape(len(ids), 12, 50).sum(axis=2, dtype=np.int64)
    contrast = np.column_stack([qsums[:, j] - qsums[:, k] for j, k in PAIRS])
    waves = [centered_features(raw[:, ix]) for ix in LAYOUTS.values()]

    source = root / "runs/u-paper-groups-01"
    reference = read_json(source / "results.json")["results"]
    fold_file = read_json(source / "folds.json")
    equal(keys.tolist(), fold_file["group_ids"], "condition tuples")
    archived_folds = index_rows(fold_file["folds"], ("mode", "target_id"))
    first = [int(ids[groups == gid].min()) for gid in range(len(keys))]
    last = [int(ids[groups == gid].max()) for gid in range(len(keys))]

    # Reuse exact per-condition ceilings across evaluation schemes.
    oracle = {}
    for g in range(len(keys)):
        subset = groups == g
        for radius in RADII_MILLI:
            diameter = 2 * SCALE * radius // 1000
            oracle[g, radius] = [window_ceiling(qsums[subset, j], grades[subset], diameter)
                                 for j in range(12)]

    predictions, group_rows, fold_logs, table, recalls = [], [], [], [], []
    by_mode = {}
    for mode in MODES:
        archived = reference[mode]
        eligible = []
        pred = np.full((len(ids), len(METHODS)), -1, dtype=np.int8)
        group_matrices = []
        for g, key in enumerate(keys):
            donors = [i for i in range(len(keys)) if i != g and
                      (mode == "condition_logo" or last[i] < first[g])]
            if len(donors) < 8:
                continue
            fold, result = compute_fold(mode, g, donors, ids, groups, grades, contrast, pressure, waves)
            fold["test_group"] = key.tolist()
            fold["methods"] = list(METHODS[:3])
            expected = archived_folds[mode, g]
            for field in fold:
                if field == "cv_scores_by_lexicographic_pair":
                    close(fold[field], expected[field], f"{mode} {g}: candidate scores")
                else:
                    equal(fold[field], expected[field], f"{mode} {g}: {field}")
            fold_logs.append(fold)
            mask = groups == g
            target_rows = np.flatnonzero(mask)
            pred[mask] = result
            matrices = np.zeros((len(METHODS), 3, 3), dtype=np.int64)
            for j in range(len(METHODS)):
                np.add.at(matrices[j], (grades[mask], result[:, j]), 1)
            rates = matrices / matrices.sum(axis=2, keepdims=True)
            group_matrices.append(rates)
            group_record = {"mode": mode, "group_id": g, **dict(zip(CONDITION_FIELDS, key.tolist())),
                            "n": len(target_rows), "first_row": first[g], "last_row": last[g]}
            for radius, name in zip(RADII_MILLI, ("U1_B0", "U1_B010", "U1_B030")):
                fractions = oracle[g, radius]
                published_cells = archived["upper_by_b_milli"][str(radius)]["grid_fraction"][len(eligible)]
                equal([str(x) for x in fractions], published_cells, f"{mode} {g}: oracle at {radius}")
                group_record[name] = float(max(fractions))
            group_record.update(zip(METHODS, np.trace(rates, axis1=1, axis2=2) / 3))
            group_rows.append(group_record)
            eligible.append(g)
        equal(eligible, archived["evaluated_group_ids"], f"{mode}: evaluated groups")
        available = np.isin(groups, eligible)
        equal(int(available.sum()), archived["n_rows"], f"{mode}: evaluated cycles")
        for i in np.flatnonzero(available):
            predictions.append({"mode": mode, "row": int(ids[i]), "prior_role": "fit" if fit[i] else "pilot",
                                "group_id": int(groups[i]), **dict(zip(CONDITION_FIELDS, keys[groups[i]].tolist())),
                                "grade": int(grades[i]), **dict(zip(METHODS, pred[i].tolist()))})
        upper = sum(max(oracle[g, 300]) for g in eligible) / len(eligible)
        equal(str(upper), archived["upper_by_b_milli"]["300"]["exact"], f"{mode}: panel oracle at .30")
        table_row = {"mode": mode, "conditions": len(eligible), "cycles": int(available.sum()),
                     "U1_B030": float(upper)}
        mean_rates = np.mean(group_matrices, axis=0)
        for j, method in enumerate(METHODS):
            raw_counts = np.zeros((3, 3), dtype=np.int64)
            np.add.at(raw_counts, (grades[available], pred[available, j]), 1)
            expected = archived["methods"][method]
            equal(raw_counts.tolist(), expected["confusion_raw"], f"{mode} {method}: counts")
            close(mean_rates[j], expected["confusion_row_proportions_group_mean"], f"{mode} {method}: grade rates")
            score = float(np.trace(mean_rates[j]) / 3)
            close(score, expected["score"], f"{mode} {method}: panel accuracy")
            table_row[method] = score
            recalls.append({"mode": mode, "method": method, **dict(zip(("none", "weak", "severe"),
                           np.diag(mean_rates[j]).tolist())), **dict(zip(("n_none", "n_weak", "n_severe"),
                           raw_counts.sum(axis=1).tolist()))})
        table.append(table_row)
        by_mode[mode] = {"groups": len(eligible), "rows": int(available.sum())}

    equal(len(fold_logs), len(archived_folds), "all reconstructed folds")
    expected_rows = read_csv(source / "predictions.csv")
    equal([{k: str(v) for k, v in row.items()} for row in predictions], expected_rows,
          "every row id, role, condition, grade and all five model predictions")
    expected_groups = read_csv(source / "groups.csv")
    equal(len(group_rows), len(expected_groups), "group table row count")
    for row, expected in zip(group_rows, expected_groups):
        equal(row["mode"], expected["mode"], "group table mode")
        for k in row.keys() - {"mode"}:
            close(row[k], float(expected[k]), f"{row['mode']} {row['group_id']}: {k}")

    printed_table = manuscript_table(root, table)
    receipt = {"status": "verified", "modes": by_mode, "verified_folds": len(fold_logs),
               "inner_candidate_scores_verified": len(fold_logs) * len(PAIRS),
               "waveform_forests_refitted": len(fold_logs) * len(waves),
               "verified_predictions": len(predictions) * len(METHODS),
               "unique_oracle_cells_verified": len(keys) * 12 * len(RADII_MILLI),
               "oracle_radii_L_min": [radius/1000 for radius in RADII_MILLI],
               "numpy": np.__version__, "scikit_learn": sklearn.__version__,
               "elapsed_seconds": round(perf_counter() - clock, 3)}
    out.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("predictions.csv", predictions), ("groups.csv", group_rows),
                           ("validation_scores.csv", printed_table), ("grade_recall.csv", recalls)):
        write_csv(out / filename, rows)
    for filename, item in (("folds.json", {"group_ids": keys.tolist(), "folds": fold_logs}),
                           ("verification.json", receipt)):
        (out / filename).write_text(json.dumps(item, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"verification": receipt, "validation_scores": printed_table, "grade_recall": recalls}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="ZeMA data.zip or its UCI wrapper ZIP")
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    parser.add_argument("--package-root", type=Path, default=ROOT, help="directory containing cycle_split.csv and runs/")
    options = parser.parse_args()
    replay(options.archive, options.package_root, options.out)
