# doPlan language re-ranking: results

lambda = 2.0; gate opens when plausible modes (score >= 0.05) spread > 4.0 m at the endpoint or > 20.0 deg in heading; compliance threshold 1.0 m.

| variant | subset | pairs with future | moments | top-1 ADE before | after | diff [95% CI] | minADE | top-1 FDE before | after |
|---|---|---|---|---|---|---|---|---|---|
| classifier | gated | 63621 | 2835 | 1.694 | 1.973 | 0.279 [0.133, 0.452] | 0.674 | 5.114 | 5.913 |
| classifier | all | 77127 | 3373 | 1.584 | 1.814 | 0.230 [0.114, 0.369] | 0.655 | 4.710 | 5.369 |
| classifier | gated_disagree | 1439 | 273 | 1.702 | 1.860 | 0.158 [0.076, 0.266] | 0.583 | 5.255 | 5.696 |
| rules_only | gated | 63621 | 2835 | 1.694 | 1.984 | 0.291 [0.143, 0.460] | 0.674 | 5.114 | 5.949 |
| rules_only | all | 77127 | 3373 | 1.584 | 1.824 | 0.240 [0.125, 0.378] | 0.655 | 4.710 | 5.399 |

| variant | subset | n | before | after | diff [95% CI] | counterfactual |
|---|---|---|---|---|---|---|
| classifier | gated | 13952 | 0.695 | 0.839 | 0.144 [0.092, 0.205] | 0.153 |
| classifier | all | 16766 | 0.679 | 0.799 | 0.120 [0.072, 0.172] | 0.127 |
| classifier | gated_disagree | 385 | 0.912 | 0.886 | -0.026 [-0.045, -0.010] | 0.008 |
| rules_only | gated | 13952 | 0.695 | 0.841 | 0.146 [0.094, 0.207] | 0.153 |
| rules_only | all | 16766 | 0.679 | 0.800 | 0.122 [0.074, 0.174] | 0.127 |

(gated_disagree is empty for: rules_only; for rules_only this is by construction, since its intents are the rule labels.)

77127 pairs over 11470 ego predictions, 3373 distinct 0.5 s moments and 90 annotations (CIs resample moments); gate open on 0.825 of pairs; classifier intent differs from the rule label on 0.022 of pairs ('gated_disagree' rows).

compliance population: rule-labelled lateral instructions with a full 6 s recorded future, counted identically for every variant.

pairs per log: 2021.08.17.16.57.11_veh-08_01200_01636 4954, 2021.08.17.17.17.01_veh-45_02314_02798 5041, 2021.08.17.18.54.02_veh-45_00665_01065 5787, 2021.10.11.02.57.41_veh-50_00352_00535 44487, 2021.10.11.02.57.41_veh-50_01522_02088 5516, 2021.10.11.08.31.07_veh-50_01750_01948 11342

Counterfactual consistency is meaningful on gated pairs only: a closed gate leaves cf_top1 == new_top1.
