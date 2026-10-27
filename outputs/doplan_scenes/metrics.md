# doPlan language re-ranking: results

lambda = 1.0; gate opens when plausible modes (score >= 0.05) spread > 4.0 m at the endpoint or > 20.0 deg in heading; compliance threshold 1.0 m; compatibility from map.

| variant | subset | pairs with future | moments | top-1 ADE before | after | diff [95% CI] | minADE | top-1 FDE before | after |
|---|---|---|---|---|---|---|---|---|---|
| classifier | gated | 180 | 37 | 2.367 | 2.377 | 0.010 [-0.123, 0.146] | 1.043 | 6.441 | 6.570 |
| classifier | all | 197 | 40 | 2.485 | 2.494 | 0.009 [-0.117, 0.131] | 1.050 | 6.963 | 7.081 |
| classifier | gated_disagree | 36 | 33 | 2.187 | 2.225 | 0.038 [-0.564, 0.630] | 0.977 | 5.935 | 6.542 |
| rules_only | gated | 180 | 37 | 2.367 | 2.369 | 0.002 [-0.043, 0.042] | 1.043 | 6.441 | 6.449 |
| rules_only | all | 197 | 40 | 2.485 | 2.487 | 0.002 [-0.040, 0.039] | 1.050 | 6.963 | 6.970 |
| classifier_geometry | gated | 180 | 37 | 2.367 | 2.185 | -0.182 [-0.351, -0.046] | 1.043 | 6.441 | 5.819 |
| classifier_geometry | all | 197 | 40 | 2.485 | 2.319 | -0.166 [-0.317, -0.042] | 1.050 | 6.963 | 6.394 |
| classifier_geometry | gated_disagree | 36 | 33 | 2.187 | 2.161 | -0.026 [-0.079, 0.000] | 0.977 | 5.935 | 5.787 |

| variant | subset | n | before | after | diff [95% CI] | counterfactual |
|---|---|---|---|---|---|---|
| classifier | gated | 30 | 0.967 | 1.000 | 0.033 [0.000, 0.120] | 0.000 |
| classifier | all | 30 | 0.967 | 1.000 | 0.033 [0.000, 0.120] | 0.000 |
| classifier | gated_disagree | 0 | n/a | n/a | n/a [n/a, n/a] | n/a |
| rules_only | gated | 30 | 0.967 | 1.000 | 0.033 [0.000, 0.120] | 0.000 |
| rules_only | all | 30 | 0.967 | 1.000 | 0.033 [0.000, 0.120] | 0.000 |
| classifier_geometry | gated | 30 | 0.967 | 1.000 | 0.033 [0.000, 0.120] | 0.000 |
| classifier_geometry | all | 30 | 0.967 | 1.000 | 0.033 [0.000, 0.120] | 0.000 |
| classifier_geometry | gated_disagree | 0 | n/a | n/a | n/a [n/a, n/a] | n/a |

Lane compliance (population: lateral instructions whose recorded future, by the lane graph, performs the instructed manoeuvre within 6 s; success: the top-1 does too):

| variant | subset | n | before | after | diff [95% CI] | counterfactual |
|---|---|---|---|---|---|---|
| classifier | gated | 16 | 0.938 | 1.000 | 0.062 [0.000, 0.231] | 0.000 |
| classifier | all | 16 | 0.938 | 1.000 | 0.062 [0.000, 0.231] | 0.000 |
| classifier | gated_disagree | 0 | n/a | n/a | n/a [n/a, n/a] | n/a |
| rules_only | gated | 16 | 0.938 | 1.000 | 0.062 [0.000, 0.231] | 0.000 |
| rules_only | all | 16 | 0.938 | 1.000 | 0.062 [0.000, 0.231] | 0.000 |
| classifier_geometry | gated | 16 | 0.938 | 1.000 | 0.062 [0.000, 0.231] | 0.000 |
| classifier_geometry | all | 16 | 0.938 | 1.000 | 0.062 [0.000, 0.231] | 0.000 |
| classifier_geometry | gated_disagree | 0 | n/a | n/a | n/a [n/a, n/a] | n/a |

(gated_disagree is empty for: rules_only; for rules_only this is by construction, since its intents are the rule labels.)

197 pairs over 40 ego predictions, 40 distinct 0.5 s moments and 7 annotations (CIs resample moments); gate open on 0.914 of pairs; classifier intent differs from the rule label on 0.198 of pairs ('gated_disagree' rows).

compliance population: rule-labelled lateral instructions with a full 6 s recorded future, counted identically for every variant.

pairs per log: 2021.07.16.21.17.55_veh-26_03772_03842 197

lane labels cover 0.971 of (pair, mode) rows; recorded-future labels: straight 171, turn_right 26

Counterfactual consistency is meaningful on gated pairs only: a closed gate leaves cf_top1 == new_top1.
