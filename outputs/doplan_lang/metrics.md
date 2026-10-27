# doPlan language re-ranking: results

lambda = 1.0; gate opens when plausible modes (score >= 0.05) spread > 4.0 m at the endpoint or > 20.0 deg in heading; compliance threshold 1.0 m; compatibility from map.

| variant | subset | pairs with future | moments | top-1 ADE before | after | diff [95% CI] | minADE | top-1 FDE before | after |
|---|---|---|---|---|---|---|---|---|---|
| classifier | gated | 63621 | 2835 | 1.694 | 1.898 | 0.204 [0.063, 0.364] | 0.674 | 5.114 | 5.732 |
| classifier | all | 77127 | 3373 | 1.584 | 1.752 | 0.168 [0.055, 0.301] | 0.655 | 4.710 | 5.220 |
| classifier | gated_disagree | 1439 | 273 | 1.702 | 1.696 | -0.005 [-0.022, 0.012] | 0.583 | 5.255 | 5.242 |
| rules_only | gated | 63621 | 2835 | 1.694 | 1.908 | 0.214 [0.074, 0.376] | 0.674 | 5.114 | 5.770 |
| rules_only | all | 77127 | 3373 | 1.584 | 1.761 | 0.177 [0.063, 0.310] | 0.655 | 4.710 | 5.251 |
| classifier_geometry | gated | 63621 | 2835 | 1.694 | 1.890 | 0.196 [0.069, 0.350] | 0.674 | 5.114 | 5.668 |
| classifier_geometry | all | 77127 | 3373 | 1.584 | 1.746 | 0.162 [0.055, 0.290] | 0.655 | 4.710 | 5.166 |
| classifier_geometry | gated_disagree | 1439 | 273 | 1.702 | 1.782 | 0.080 [0.044, 0.123] | 0.583 | 5.255 | 5.478 |

| variant | subset | n | before | after | diff [95% CI] | counterfactual |
|---|---|---|---|---|---|---|
| classifier | gated | 13952 | 0.695 | 0.776 | 0.081 [0.040, 0.136] | 0.049 |
| classifier | all | 16766 | 0.679 | 0.747 | 0.068 [0.031, 0.114] | 0.041 |
| classifier | gated_disagree | 385 | 0.912 | 0.912 | 0.000 [-0.008, 0.008] | 0.000 |
| rules_only | gated | 13952 | 0.695 | 0.776 | 0.082 [0.040, 0.136] | 0.050 |
| rules_only | all | 16766 | 0.679 | 0.747 | 0.068 [0.031, 0.114] | 0.041 |
| classifier_geometry | gated | 13952 | 0.695 | 0.824 | 0.130 [0.080, 0.190] | 0.120 |
| classifier_geometry | all | 16766 | 0.679 | 0.787 | 0.108 [0.062, 0.160] | 0.100 |
| classifier_geometry | gated_disagree | 385 | 0.912 | 0.906 | -0.005 [-0.013, 0.000] | 0.005 |

Lane compliance (population: lateral instructions whose recorded future, by the lane graph, performs the instructed manoeuvre within 6 s; success: the top-1 does too):

| variant | subset | n | before | after | diff [95% CI] | counterfactual |
|---|---|---|---|---|---|---|
| classifier | gated | 6360 | 0.686 | 0.895 | 0.210 [0.120, 0.314] | 0.010 |
| classifier | all | 7403 | 0.702 | 0.883 | 0.180 [0.102, 0.276] | 0.008 |
| classifier | gated_disagree | 116 | 0.776 | 0.741 | -0.034 [-0.082, 0.000] | 0.000 |
| rules_only | gated | 6360 | 0.686 | 0.901 | 0.215 [0.125, 0.322] | 0.010 |
| rules_only | all | 7403 | 0.702 | 0.887 | 0.185 [0.108, 0.280] | 0.009 |
| classifier_geometry | gated | 6360 | 0.686 | 0.849 | 0.164 [0.085, 0.255] | 0.009 |
| classifier_geometry | all | 7403 | 0.702 | 0.843 | 0.141 [0.072, 0.222] | 0.008 |
| classifier_geometry | gated_disagree | 116 | 0.776 | 0.784 | 0.009 [0.000, 0.027] | 0.000 |

(gated_disagree is empty for: rules_only; for rules_only this is by construction, since its intents are the rule labels.)

77127 pairs over 11470 ego predictions, 3373 distinct 0.5 s moments and 90 annotations (CIs resample moments); gate open on 0.825 of pairs; classifier intent differs from the rule label on 0.022 of pairs ('gated_disagree' rows).

compliance population: rule-labelled lateral instructions with a full 6 s recorded future, counted identically for every variant.

pairs per log: 2021.08.17.16.57.11_veh-08_01200_01636 4954, 2021.08.17.17.17.01_veh-45_02314_02798 5041, 2021.08.17.18.54.02_veh-45_00665_01065 5787, 2021.10.11.02.57.41_veh-50_00352_00535 44487, 2021.10.11.02.57.41_veh-50_01522_02088 5516, 2021.10.11.08.31.07_veh-50_01750_01948 11342

lane labels cover 0.823 of (pair, mode) rows; recorded-future labels: straight 53360, turn_left 10385, turn_right 7153, None 5366, lane_right 516, lane_left 347

Counterfactual consistency is meaningful on gated pairs only: a closed gate leaves cf_top1 == new_top1.
