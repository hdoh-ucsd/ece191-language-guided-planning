# Embedding-space probe: left vs right, and representative commands

## 1. Word level

- cosine(left, right) = 0.590; `right` is neighbour #1 of `left`, `left` is neighbour #1 of `right`
- |v(left)| = 1.98, |v(right)| = 2.10, |v(left) - v(right)| = 1.85

## 2. Clause level (training first-action clauses, rule labels)

- direction axis d = unit(v(left) - v(right)); cosine with the turn_left/turn_right centroid difference = 0.975
- turn_left vs turn_right along d: AUC 0.996 (means +0.222 / -0.164, pooled sd 0.130)
- lane_left vs lane_right along the same d: AUC 0.996
- turn_left vs turn_right along the centroid axis: AUC 0.991
- projection on d by intent (mean, sd): turn_left +0.222 (0.097), turn_right -0.164 (0.156), lane_left +0.046 (0.063), lane_right -0.266 (0.090), stop +0.104 (0.106), straight +0.027 (0.107), other -0.047 (0.124)
- words with the largest projection on d, left pole: [('trailer', 1.14), ('an', 1.05), ('out', 0.93), ('left', 0.79), ('or', 0.78), ('stationary', 0.76), ('floor', 0.72), ('direction', 0.7)]; right pole: [('right', -1.06), ('balloon', -1.0), ('west', -0.81), ('righ', -0.8), ('sedan', -0.68), ('allow', -0.66), ('tturn', -0.66), ('limos', -0.65)]; `left` ranks #4 and `right` #1 of 616 words
- fraction of total clause variance along d: 2.942%; PCA variance ratios PC1..5 = [0.17, 0.105, 0.088, 0.079, 0.052]; |cos(d, PC1..5)| = [0.087, 0.102, 0.252, 0.262, 0.156] (most aligned PC: 13)
- parallelogram: cos(turn_left - turn_right, lane_left - lane_right) = 0.804; cos(turn_left - lane_left, turn_right - lane_right) = 0.959
- centroid cosines: turn_left/turn_right 0.964, turn_left/stop 0.865, turn_left/straight 0.756
- left vs right only: cosine silhouette 0.116; nearest-centroid accuracy 0.945; 5-fold logistic regression accuracy raw 0.992, standardised 0.996

## 3. Counterfactual left<->right swap of every turn clause

- n = 2355; cosine(original, swapped) median 0.983 [p10 0.861, p90 0.995]
- cosine(original, random clause of the same intent) median 0.782 [p10 0.621, p90 0.928]
- cosine(original, random clause of the opposite intent) median 0.750 [p10 0.592, p90 0.901]
- intent classifier follows the swap in 92.6% of 2315 clauses (93.2% of those it classified correctly before the swap); it keeps predicting the original direction in 0.4%; clauses that fail to flip average 10.9 tokens vs 6.3 for those that flip

## 4. Representative commands

| # | command | rule intent | classifier argmax (p) | P(turn_left) | P(turn_right) | proj. on d | in-vocab tokens |
|---|---|---|---|---|---|---|---|
| 1 | turn left at the intersection | turn_left | turn_left (1.00) | 1.00 | 0.00 | +0.181 | 5/5 |
| 2 | turn right at the intersection | turn_right | turn_right (1.00) | 0.00 | 1.00 | -0.189 | 5/5 |
| 3 | make a left turn at the light | turn_left | turn_left (1.00) | 1.00 | 0.00 | +0.228 | 7/7 |
| 4 | take the next right | turn_right | turn_right (1.00) | 0.00 | 1.00 | -0.235 | 4/4 |
| 5 | make a u turn | turn_left | turn_left (1.00) | 1.00 | 0.00 | +0.206 | 4/4 |
| 6 | change to the left lane | lane_left | lane_left (1.00) | 0.00 | 0.00 | +0.042 | 5/5 |
| 7 | merge into the right lane | lane_right | lane_right (1.00) | 0.00 | 0.00 | -0.380 | 5/5 |
| 8 | stop at the stop sign | stop | stop (1.00) | 0.00 | 0.00 | +0.156 | 5/5 |
| 9 | pull over here | stop | stop (1.00) | 0.00 | 0.00 | +0.054 | 3/3 |
| 10 | go straight through the intersection | straight | straight (1.00) | 0.00 | 0.00 | -0.027 | 5/5 |
| 11 | keep going straight | straight | straight (1.00) | 0.00 | 0.00 | +0.141 | 3/3 |
| 12 | stay in the right lane | straight | straight (0.96) | 0.00 | 0.00 | -0.158 | 5/5 |
| 13 | head toward the hotel | other | other (1.00) | 0.00 | 0.00 | -0.136 | 4/4 |
| 14 | go around the roundabout | other | other (0.98) | 0.00 | 0.00 | -0.166 | 4/4 |
| 15 | turn right from the left lane | other | lane_left (0.79) | 0.01 | 0.00 | +0.011 | 6/6 |

Figures: fig1_left_right.png, fig2_command_space.png, fig3_tsne_by_intent.png, fig4_command_encoding.png
