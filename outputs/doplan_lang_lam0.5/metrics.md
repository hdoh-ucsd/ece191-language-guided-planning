# doPlan language re-ranking: results

lambda = 0.5; gate opens when plausible modes (score >= 0.05) spread > 4.0 m at the endpoint or > 20.0 deg in heading; compliance threshold 1.0 m.

| variant | subset | pairs with future | moments | top-1 ADE before | after | diff [95% CI] | minADE | top-1 FDE before | after |
|---|---|---|---|---|---|---|---|---|---|
| classifier | gated | 63621 | 2835 | 1.694 | 1.788 | 0.094 [-0.008, 0.224] | 0.674 | 5.114 | 5.356 |
| classifier | all | 77127 | 3373 | 1.584 | 1.661 | 0.077 [-0.002, 0.178] | 0.655 | 4.710 | 4.910 |
| classifier | gated_disagree | 1439 | 273 | 1.702 | 1.744 | 0.042 [0.020, 0.067] | 0.583 | 5.255 | 5.376 |
| rules_only | gated | 63621 | 2835 | 1.694 | 1.792 | 0.098 [-0.001, 0.228] | 0.674 | 5.114 | 5.367 |
| rules_only | all | 77127 | 3373 | 1.584 | 1.665 | 0.081 [0.002, 0.182] | 0.655 | 4.710 | 4.919 |

| variant | subset | n | before | after | diff [95% CI] | counterfactual |
|---|---|---|---|---|---|---|
| classifier | gated | 13952 | 0.695 | 0.778 | 0.084 [0.045, 0.135] | 0.080 |
| classifier | all | 16766 | 0.679 | 0.749 | 0.070 [0.033, 0.115] | 0.066 |
| classifier | gated_disagree | 385 | 0.912 | 0.912 | 0.000 [0.000, 0.000] | 0.000 |
| rules_only | gated | 13952 | 0.695 | 0.781 | 0.086 [0.047, 0.137] | 0.081 |
| rules_only | all | 16766 | 0.679 | 0.751 | 0.072 [0.035, 0.117] | 0.067 |

(gated_disagree is empty for: rules_only; for rules_only this is by construction, since its intents are the rule labels.)

77127 pairs over 11470 ego predictions, 3373 distinct 0.5 s moments and 90 annotations (CIs resample moments); gate open on 0.825 of pairs; classifier intent differs from the rule label on 0.022 of pairs ('gated_disagree' rows).

compliance population: rule-labelled lateral instructions with a full 6 s recorded future, counted identically for every variant.

pairs per log: 2021.08.17.16.57.11_veh-08_01200_01636 4954, 2021.08.17.17.17.01_veh-45_02314_02798 5041, 2021.08.17.18.54.02_veh-45_00665_01065 5787, 2021.10.11.02.57.41_veh-50_00352_00535 44487, 2021.10.11.02.57.41_veh-50_01522_02088 5516, 2021.10.11.08.31.07_veh-50_01750_01948 11342

Counterfactual consistency is meaningful on gated pairs only: a closed gate leaves cf_top1 == new_top1.
