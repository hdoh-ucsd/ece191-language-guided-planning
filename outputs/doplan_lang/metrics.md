# doPlan language re-ranking: results

lambda = 1.0; gate opens when plausible modes (score >= 0.05) spread > 4.0 m at the endpoint or > 20.0 deg in heading; compliance threshold 1.0 m.

| variant | subset | pairs with future | moments | top-1 ADE before | after | diff [95% CI] | minADE | top-1 FDE before | after |
|---|---|---|---|---|---|---|---|---|---|
| classifier | gated | 63621 | 2835 | 1.694 | 1.890 | 0.196 [0.069, 0.350] | 0.674 | 5.114 | 5.668 |
| classifier | all | 77127 | 3373 | 1.584 | 1.746 | 0.162 [0.055, 0.290] | 0.655 | 4.710 | 5.166 |
| classifier | gated_disagree | 1439 | 273 | 1.702 | 1.782 | 0.080 [0.044, 0.123] | 0.583 | 5.255 | 5.478 |
| rules_only | gated | 63621 | 2835 | 1.694 | 1.897 | 0.203 [0.079, 0.357] | 0.674 | 5.114 | 5.688 |
| rules_only | all | 77127 | 3373 | 1.584 | 1.751 | 0.168 [0.062, 0.295] | 0.655 | 4.710 | 5.184 |

| variant | subset | n | before | after | diff [95% CI] | counterfactual |
|---|---|---|---|---|---|---|
| classifier | gated | 13952 | 0.695 | 0.824 | 0.130 [0.080, 0.190] | 0.120 |
| classifier | all | 16766 | 0.679 | 0.787 | 0.108 [0.062, 0.160] | 0.100 |
| classifier | gated_disagree | 385 | 0.912 | 0.906 | -0.005 [-0.013, 0.000] | 0.005 |
| rules_only | gated | 13952 | 0.695 | 0.827 | 0.133 [0.083, 0.193] | 0.123 |
| rules_only | all | 16766 | 0.679 | 0.789 | 0.111 [0.064, 0.163] | 0.103 |

(gated_disagree is empty for: rules_only; for rules_only this is by construction, since its intents are the rule labels.)

77127 pairs over 11470 ego predictions, 3373 distinct 0.5 s moments and 90 annotations (CIs resample moments); gate open on 0.825 of pairs; classifier intent differs from the rule label on 0.022 of pairs ('gated_disagree' rows).

compliance population: rule-labelled lateral instructions with a full 6 s recorded future, counted identically for every variant.

pairs per log: 2021.08.17.16.57.11_veh-08_01200_01636 4954, 2021.08.17.17.17.01_veh-45_02314_02798 5041, 2021.08.17.18.54.02_veh-45_00665_01065 5787, 2021.10.11.02.57.41_veh-50_00352_00535 44487, 2021.10.11.02.57.41_veh-50_01522_02088 5516, 2021.10.11.08.31.07_veh-50_01750_01948 11342

Counterfactual consistency is meaningful on gated pairs only: a closed gate leaves cf_top1 == new_top1.
