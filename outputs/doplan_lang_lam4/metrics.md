# doPlan language re-ranking: results

lambda = 4.0; gate opens when plausible modes (score >= 0.05) spread > 4.0 m at the endpoint or > 20.0 deg in heading; compliance threshold 1.0 m.

| variant | subset | pairs with future | moments | top-1 ADE before | after | diff [95% CI] | minADE | top-1 FDE before | after |
|---|---|---|---|---|---|---|---|---|---|
| classifier | gated | 63621 | 2835 | 1.694 | 2.072 | 0.378 [0.217, 0.552] | 0.674 | 5.114 | 6.222 |
| classifier | all | 77127 | 3373 | 1.584 | 1.896 | 0.312 [0.188, 0.457] | 0.655 | 4.710 | 5.624 |
| classifier | gated_disagree | 1439 | 273 | 1.702 | 1.888 | 0.187 [0.076, 0.323] | 0.583 | 5.255 | 5.803 |
| rules_only | gated | 63621 | 2835 | 1.694 | 2.087 | 0.393 [0.233, 0.565] | 0.674 | 5.114 | 6.268 |
| rules_only | all | 77127 | 3373 | 1.584 | 1.908 | 0.324 [0.201, 0.466] | 0.655 | 4.710 | 5.662 |

| variant | subset | n | before | after | diff [95% CI] | counterfactual |
|---|---|---|---|---|---|---|
| classifier | gated | 13952 | 0.695 | 0.848 | 0.153 [0.097, 0.221] | 0.165 |
| classifier | all | 16766 | 0.679 | 0.806 | 0.128 [0.077, 0.181] | 0.138 |
| classifier | gated_disagree | 385 | 0.912 | 0.875 | -0.036 [-0.063, -0.014] | 0.010 |
| rules_only | gated | 13952 | 0.695 | 0.851 | 0.156 [0.100, 0.224] | 0.165 |
| rules_only | all | 16766 | 0.679 | 0.809 | 0.130 [0.079, 0.183] | 0.138 |

(gated_disagree is empty for: rules_only; for rules_only this is by construction, since its intents are the rule labels.)

77127 pairs over 11470 ego predictions, 3373 distinct 0.5 s moments and 90 annotations (CIs resample moments); gate open on 0.825 of pairs; classifier intent differs from the rule label on 0.022 of pairs ('gated_disagree' rows).

compliance population: rule-labelled lateral instructions with a full 6 s recorded future, counted identically for every variant.

pairs per log: 2021.08.17.16.57.11_veh-08_01200_01636 4954, 2021.08.17.17.17.01_veh-45_02314_02798 5041, 2021.08.17.18.54.02_veh-45_00665_01065 5787, 2021.10.11.02.57.41_veh-50_00352_00535 44487, 2021.10.11.02.57.41_veh-50_01522_02088 5516, 2021.10.11.08.31.07_veh-50_01750_01948 11342

Counterfactual consistency is meaningful on gated pairs only: a closed gate leaves cf_top1 == new_top1.
