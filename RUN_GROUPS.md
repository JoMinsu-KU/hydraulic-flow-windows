# Refit the condition-held-out and forward-validation models

From the repository root, use the versions in `requirements.txt` and the original [ZeMA flow and pressure ZIP](https://doi.org/10.24432/C5CW21). The direct laboratory `data.zip` and UCI's ZIP containing `data.zip` are supported.

```bash
python replay_groups.py --archive /path/to/data.zip --out /path/to/new/group-replay
```

The output directory must be new. The script reads the raw FS1 and PS1 waveforms for the 849 training and 300 exploration cycles identified in `cycle_split.csv`. Cooler, valve and accumulator settings define 38 groups. The two validation schemes hold out a group at a time or train only on complete groups preceding the target group in archive order. Eight of the 38 groups have fewer than eight eligible earlier groups, leaving 30 forward evaluations.

In each outer training fold, four inner group splits evaluate all 66 distinct 5-s window pairs. A tied inner score is resolved by the mean pressure contrast in the training data, then by lexicographic pair order. The script refits the grade medians and both 96-tree waveform forests. It reconstructs the folds before comparing them with the public fold assignments. All 2,058 row predictions for each of five methods, all 68 sets of training medians, and all 4,488 candidate scores must match. The single-mean ceiling is recalculated for all 38 groups and 12 windows at the three offset half-widths 0, 0.10 and 0.30 L/min using integer sums, class weights and closed-interval conflicts.

| Output | Contents |
|---|---|
| `validation_scores.csv` | Paper table in percent. The 38- and 30-condition columns come from the new fits. The separate 10-condition column uses the original audited held-out counts. Blank cells mark methods outside that frozen evaluation. |
| `grade_recall.csv` | Mean within-condition recall for each grade as a fraction and the total cycle count per grade. Groups, then grades, carry equal weight in the reported balanced accuracy. |
| `predictions.csv`, `groups.csv` | Cycle identifiers and predicted grades for all five methods, and condition-level accuracies and ceilings. |
| `folds.json` | Outer and inner group memberships, 66 inner scores, chosen pairs and twice the integer median for each grade and scalar method. Divide this last quantity by 100,000,000 to recover L/min. |
| `verification.json` | Number of predictions, folds and oracle cells checked, library versions and elapsed computation time. |

For a separate raw-data reconstruction of the frozen 10-condition confirmation, including its fixed two-mean count, single-mean ceiling and exact 85% crossing, use `python replay_primary.py --archive /path/to/data.zip`. The group refits assess conditions within the same ZeMA test rig.
