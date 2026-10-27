# scene7

- clip: `2021.07.16.21.17.55_veh-26_03772_03842`
- start timestamp: 1626474267522457 us = 3.60 s after the first lidar frame of the log (doPlan frame 18.0)
- moments found: 21 (every 0.5 s); max |dt| to the nominal moment 4 ms

## Instructions whose window covers the scene start

| annotation | window (s after log start) | instruction | rule intent |
|---|---|---|---|
| doplan_000013 | 3.2 to 45.0 | get behind the black suv at the stoplight | other |
| doplan_003808 | 0.2 to 67.0 | Take a right then go into the second leftmost lane. | turn_right |
| doplan_004144 | 3.2 to 65.0 | Switch into the second right most lane. | lane_right |
| doplan_006457 | 2.2 to 65.4 | Take a right at the stop sign. Go one lane to the left. | turn_right |

## Original vs re-ranked top-1 path

Re-ranking changed the top-1 on 12 of 98 (moment, instruction) rows (12 of 21 moments). The changed top-1 is closer to the recorded path (lower ADE) on 4 rows and farther on 8.

Features are in the ego frame at the prediction time: lat = endpoint lateral offset (m, + is left), dpsi = final heading change (deg, + is left), path = length (m) over 6 s.

| t (s) | annotation | intent | mode orig -> new | original top-1 (lat, dpsi, path) | re-ranked top-1 (lat, dpsi, path) | recorded future (lat, dpsi, path) | lane labels orig -> new (recorded) | ADE orig -> new (m) | FDE orig -> new (m) | closer? |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | doplan_000013 | stop | 0 -> 3 | -19.9, -73, 25 | -8.3, -100, 14 | -19.9, -71, 25 | turn_right -> turn_right (turn_right) | 1.13 -> 3.23 | 0.91 -> 13.05 | no |
| 1.0 | doplan_000013 | stop | 0 -> 3 | -23.0, -69, 28 | -5.2, +165, 9 | -21.8, -67, 26 | turn_right -> turn_right (turn_right) | 1.17 -> 5.45 | 1.16 -> 18.74 | no |
| 1.5 | doplan_000013 | stop | 0 -> 4 | -19.4, -63, 25 | -7.3, +175, 13 | -22.3, -59, 28 | turn_right -> - (turn_right) | 0.94 -> 5.25 | 3.43 -> 18.63 | no |
| 2.0 | doplan_000013 | stop | 0 -> 3 | -23.3, -47, 32 | -7.2, +165, 14 | -21.0, -48, 30 | turn_right -> - (turn_right) | 1.22 -> 5.93 | 2.58 -> 20.28 | no |
| 2.5 | doplan_000013 | stop | 0 -> 3 | -22.8, -31, 39 | -11.2, -144, 19 | -17.1, -35, 31 | straight -> straight (straight) | 3.10 -> 3.72 | 7.57 -> 14.98 | no |
| 3.5 | doplan_000013 | stop | 0 -> 4 | -4.7, -8, 34 | -3.5, -18, 23 | -4.2, -7, 34 | straight -> straight (straight) | 0.65 -> 2.55 | 0.63 -> 11.04 | no |
| 4.5 | doplan_001238 | lane_left | 0 -> 1 | +0.6, +0, 34 | +2.0, +1, 39 | -0.1, -1, 34 | straight -> lane_left (straight) | 0.73 -> 1.92 | 0.79 -> 5.79 | no |
| 5.0 | doplan_000013 | stop | 0 -> 1 | +2.1, +1, 39 | +0.6, -1, 34 | -0.2, +0, 33 | lane_left -> straight (straight) | 1.91 -> 0.99 | 6.23 -> 0.93 | yes |
| 6.0 | doplan_001238 | lane_left | 0 -> 2 | +1.1, +0, 39 | +2.0, +1, 45 | -0.2, +1, 30 | straight -> lane_left (straight) | 2.37 -> 4.56 | 8.80 -> 14.95 | no |
| 8.5 | doplan_000013 | stop | 0 -> 3 | -0.5, -1, 35 | -0.4, -1, 26 | -0.0, -1, 17 | straight -> straight (straight) | 5.74 -> 3.54 | 17.44 -> 6.93 | yes |
| 9.5 | doplan_000013 | stop | 0 -> 1 | -0.5, -0, 29 | -0.2, -1, 12 | +0.0, +3, 13 | straight -> straight (straight) | 5.37 -> 0.46 | 15.55 -> 1.32 | yes |
| 10.0 | doplan_000013 | stop | 0 -> 4 | -0.3, -1, 23 | -0.1, +2, 12 | +0.0, -0, 12 | straight -> straight (straight) | 4.81 -> 1.44 | 11.63 -> 0.62 | yes |

![scene7_paths](scene7_paths.png)
![scene7_changed](scene7_changed.png)

## Moments

| t (s) | annotation | intent | gate | MTR scores | fused scores | top-1 MTR -> fused | labels MTR top-1 / fused / recorded | ADE top-1 MTR / fused (m) | minADE |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | doplan_000013 | stop | open | 0.55 0.19 0.18 0.07 0.01 | 0.48 0.17 0.16 0.06 0.15 | 0 -> 0 | turn_right / turn_right / turn_right | 2.05 / 2.05 | 1.36 |
| 0.0 | doplan_003808 | turn_right | open | 0.55 0.19 0.18 0.07 0.01 | 0.56 0.19 0.18 0.07 0.00 | 0 -> 0 | turn_right / turn_right / turn_right | 2.05 / 2.05 | 1.36 |
| 0.0 | doplan_004144 | lane_right | open | 0.55 0.19 0.18 0.07 0.01 | 0.55 0.19 0.18 0.07 0.01 | 0 -> 0 | turn_right / turn_right / turn_right | 2.05 / 2.05 | 1.36 |
| 0.0 | doplan_006457 | turn_right | open | 0.55 0.19 0.18 0.07 0.01 | 0.56 0.19 0.18 0.07 0.00 | 0 -> 0 | turn_right / turn_right / turn_right | 2.05 / 2.05 | 1.36 |
| 0.5 | doplan_000013 | stop | open | 0.51 0.22 0.11 0.09 0.06 | 0.13 0.05 0.03 0.48 0.31 | 0 -> 3 | turn_right / turn_right+stop / turn_right | 1.13 / 3.23 | 1.13 |
| 0.5 | doplan_003808 | turn_right | open | 0.51 0.22 0.11 0.09 0.06 | 0.55 0.23 0.12 0.10 0.00 | 0 -> 0 | turn_right / turn_right / turn_right | 1.13 / 1.13 | 1.13 |
| 0.5 | doplan_004144 | lane_right | open | 0.51 0.22 0.11 0.09 0.06 | 0.51 0.22 0.11 0.09 0.06 | 0 -> 0 | turn_right / turn_right / turn_right | 1.13 / 1.13 | 1.13 |
| 0.5 | doplan_006457 | turn_right | open | 0.51 0.22 0.11 0.09 0.06 | 0.55 0.23 0.12 0.10 0.00 | 0 -> 0 | turn_right / turn_right / turn_right | 1.13 / 1.13 | 1.13 |
| 1.0 | doplan_000013 | stop | open | 0.37 0.36 0.21 0.04 0.02 | 0.17 0.16 0.10 0.41 0.16 | 0 -> 3 | turn_right / turn_right+stop / turn_right | 1.17 / 5.45 | 1.17 |
| 1.0 | doplan_003808 | turn_right | open | 0.37 0.36 0.21 0.04 0.02 | 0.38 0.37 0.21 0.04 0.00 | 0 -> 0 | turn_right / turn_right / turn_right | 1.17 / 1.17 | 1.17 |
| 1.0 | doplan_004144 | lane_right | open | 0.37 0.36 0.21 0.04 0.02 | 0.37 0.36 0.21 0.04 0.02 | 0 -> 0 | turn_right / turn_right / turn_right | 1.17 / 1.17 | 1.17 |
| 1.0 | doplan_006457 | turn_right | open | 0.37 0.36 0.21 0.04 0.02 | 0.38 0.37 0.21 0.04 0.00 | 0 -> 0 | turn_right / turn_right / turn_right | 1.17 / 1.17 | 1.17 |
| 1.5 | doplan_000013 | stop | open | 0.37 0.31 0.16 0.12 0.05 | 0.18 0.15 0.08 0.06 0.54 | 0 -> 4 | turn_right / -+stop / turn_right | 0.94 / 5.25 | 0.94 |
| 1.5 | doplan_003808 | turn_right | open | 0.37 0.31 0.16 0.12 0.05 | 0.39 0.32 0.17 0.12 0.00 | 0 -> 0 | turn_right / turn_right / turn_right | 0.94 / 0.94 | 0.94 |
| 1.5 | doplan_004144 | lane_right | open | 0.37 0.31 0.16 0.12 0.05 | 0.37 0.31 0.16 0.12 0.05 | 0 -> 0 | turn_right / turn_right / turn_right | 0.94 / 0.94 | 0.94 |
| 1.5 | doplan_006457 | turn_right | open | 0.37 0.31 0.16 0.12 0.05 | 0.39 0.32 0.17 0.12 0.00 | 0 -> 0 | turn_right / turn_right / turn_right | 0.94 / 0.94 | 0.94 |
| 2.0 | doplan_000013 | stop | open | 0.46 0.39 0.07 0.06 0.02 | 0.20 0.18 0.03 0.58 0.01 | 0 -> 3 | turn_right / -+stop / turn_right | 1.22 / 5.93 | 1.22 |
| 2.0 | doplan_003808 | turn_right | open | 0.46 0.39 0.07 0.06 0.02 | 0.49 0.42 0.07 0.00 0.02 | 0 -> 0 | turn_right / turn_right / turn_right | 1.22 / 1.22 | 1.22 |
| 2.0 | doplan_004144 | lane_right | open | 0.46 0.39 0.07 0.06 0.02 | 0.46 0.39 0.07 0.06 0.02 | 0 -> 0 | turn_right / turn_right / turn_right | 1.22 / 1.22 | 1.22 |
| 2.0 | doplan_006457 | turn_right | open | 0.46 0.39 0.07 0.06 0.02 | 0.49 0.42 0.07 0.00 0.02 | 0 -> 0 | turn_right / turn_right / turn_right | 1.22 / 1.22 | 1.22 |
| 2.5 | doplan_000013 | stop | open | 0.45 0.20 0.19 0.09 0.06 | 0.16 0.07 0.07 0.68 0.02 | 0 -> 3 | straight / straight+stop / straight | 3.10 / 3.72 | 1.65 |
| 2.5 | doplan_003808 | turn_right | open | 0.45 0.20 0.19 0.09 0.06 | 0.45 0.20 0.19 0.09 0.06 | 0 -> 0 | straight / straight / straight | 3.10 / 3.10 | 1.65 |
| 2.5 | doplan_004144 | lane_right | open | 0.45 0.20 0.19 0.09 0.06 | 0.45 0.20 0.19 0.09 0.06 | 0 -> 0 | straight / straight / straight | 3.10 / 3.10 | 1.65 |
| 2.5 | doplan_006457 | turn_right | open | 0.45 0.20 0.19 0.09 0.06 | 0.45 0.20 0.19 0.09 0.06 | 0 -> 0 | straight / straight / straight | 3.10 / 3.10 | 1.65 |
| 3.0 | doplan_000013 | stop | open | 0.67 0.18 0.13 0.02 0.01 | 0.44 0.12 0.08 0.28 0.08 | 0 -> 0 | straight / straight / straight | 0.73 / 0.73 | 0.73 |
| 3.0 | doplan_003808 | turn_right | open | 0.67 0.18 0.13 0.02 0.01 | 0.67 0.18 0.13 0.02 0.01 | 0 -> 0 | straight / straight / straight | 0.73 / 0.73 | 0.73 |
| 3.0 | doplan_004144 | lane_right | open | 0.67 0.18 0.13 0.02 0.01 | 0.67 0.18 0.13 0.02 0.01 | 0 -> 0 | straight / straight / straight | 0.73 / 0.73 | 0.73 |
| 3.0 | doplan_006457 | turn_right | open | 0.67 0.18 0.13 0.02 0.01 | 0.67 0.18 0.13 0.02 0.01 | 0 -> 0 | straight / straight / straight | 0.73 / 0.73 | 0.73 |
| 3.5 | doplan_000013 | stop | open | 0.50 0.30 0.10 0.07 0.03 | 0.32 0.19 0.07 0.04 0.39 | 0 -> 4 | straight / straight+stop / straight | 0.65 / 2.55 | 0.65 |
| 3.5 | doplan_003808 | turn_right | open | 0.50 0.30 0.10 0.07 0.03 | 0.50 0.30 0.10 0.07 0.03 | 0 -> 0 | straight / straight / straight | 0.65 / 0.65 | 0.65 |
| 3.5 | doplan_004144 | lane_right | open | 0.50 0.30 0.10 0.07 0.03 | 0.50 0.30 0.10 0.07 0.03 | 0 -> 0 | straight / straight / straight | 0.65 / 0.65 | 0.65 |
| 3.5 | doplan_006457 | turn_right | open | 0.50 0.30 0.10 0.07 0.03 | 0.50 0.30 0.10 0.07 0.03 | 0 -> 0 | straight / straight / straight | 0.65 / 0.65 | 0.65 |
| 4.0 | doplan_000013 | stop | open | 0.49 0.42 0.08 0.01 0.01 | 0.49 0.41 0.09 0.01 0.01 | 0 -> 0 | lane_left / lane_left / straight | 2.30 / 2.30 | 1.21 |
| 4.0 | doplan_001238 | lane_left | open | 0.49 0.42 0.08 0.01 0.01 | 0.54 0.45 0.00 0.00 0.01 | 0 -> 0 | lane_left / lane_left / straight | 2.30 / 2.30 | 1.21 |
| 4.0 | doplan_003808 | turn_right | open | 0.49 0.42 0.08 0.01 0.01 | 0.49 0.42 0.08 0.01 0.01 | 0 -> 0 | lane_left / lane_left / straight | 2.30 / 2.30 | 1.21 |
| 4.0 | doplan_004144 | lane_right | open | 0.49 0.42 0.08 0.01 0.01 | 0.49 0.42 0.08 0.01 0.01 | 0 -> 0 | lane_left / lane_left / straight | 2.30 / 2.30 | 1.21 |
| 4.0 | doplan_006457 | turn_right | open | 0.49 0.42 0.08 0.01 0.01 | 0.49 0.42 0.08 0.01 0.01 | 0 -> 0 | lane_left / lane_left / straight | 2.30 / 2.30 | 1.21 |
| 4.5 | doplan_000013 | stop | open | 0.57 0.33 0.08 0.01 0.01 | 0.59 0.31 0.08 0.01 0.01 | 0 -> 0 | straight / straight / straight | 0.73 / 0.73 | 0.73 |
| 4.5 | doplan_001238 | lane_left | open | 0.57 0.33 0.08 0.01 0.01 | 0.04 0.94 0.01 0.00 0.02 | 0 -> 1 | straight / lane_left / straight | 0.73 / 1.92 | 0.73 |
| 4.5 | doplan_003808 | turn_right | open | 0.57 0.33 0.08 0.01 0.01 | 0.57 0.33 0.08 0.01 0.01 | 0 -> 0 | straight / straight / straight | 0.73 / 0.73 | 0.73 |
| 4.5 | doplan_004144 | lane_right | open | 0.57 0.33 0.08 0.01 0.01 | 0.57 0.33 0.08 0.01 0.01 | 0 -> 0 | straight / straight / straight | 0.73 / 0.73 | 0.73 |
| 4.5 | doplan_006457 | turn_right | open | 0.57 0.33 0.08 0.01 0.01 | 0.57 0.33 0.08 0.01 0.01 | 0 -> 0 | straight / straight / straight | 0.73 / 0.73 | 0.73 |
| 5.0 | doplan_000013 | stop | open | 0.47 0.43 0.07 0.02 0.01 | 0.45 0.45 0.07 0.01 0.01 | 0 -> 1 | lane_left / straight / straight | 1.91 / 0.99 | 0.99 |
| 5.0 | doplan_001238 | lane_left | open | 0.47 0.43 0.07 0.02 0.01 | 0.95 0.02 0.00 0.03 0.00 | 0 -> 0 | lane_left / lane_left / straight | 1.91 / 1.91 | 0.99 |
| 5.0 | doplan_003808 | turn_right | open | 0.47 0.43 0.07 0.02 0.01 | 0.47 0.43 0.07 0.02 0.01 | 0 -> 0 | lane_left / lane_left / straight | 1.91 / 1.91 | 0.99 |
| 5.0 | doplan_004144 | lane_right | open | 0.47 0.43 0.07 0.02 0.01 | 0.48 0.43 0.07 0.02 0.01 | 0 -> 0 | lane_left / lane_left / straight | 1.91 / 1.91 | 0.99 |
| 5.0 | doplan_006457 | turn_right | open | 0.47 0.43 0.07 0.02 0.01 | 0.47 0.43 0.07 0.02 0.01 | 0 -> 0 | lane_left / lane_left / straight | 1.91 / 1.91 | 0.99 |
| 5.5 | doplan_000013 | stop | open | 0.63 0.24 0.07 0.05 0.01 | 0.49 0.21 0.05 0.04 0.21 | 0 -> 0 | lane_left / lane_left / straight | 2.42 / 2.42 | 0.71 |
| 5.5 | doplan_001238 | lane_left | open | 0.63 0.24 0.07 0.05 0.01 | 0.89 0.01 0.10 0.00 0.00 | 0 -> 0 | lane_left / lane_left / straight | 2.42 / 2.42 | 0.71 |
| 5.5 | doplan_003808 | turn_right | open | 0.63 0.24 0.07 0.05 0.01 | 0.63 0.24 0.07 0.05 0.01 | 0 -> 0 | lane_left / lane_left / straight | 2.42 / 2.42 | 0.71 |
| 5.5 | doplan_004144 | lane_right | open | 0.63 0.24 0.07 0.05 0.01 | 0.63 0.24 0.07 0.05 0.01 | 0 -> 0 | lane_left / lane_left / straight | 2.42 / 2.42 | 0.71 |
| 5.5 | doplan_006457 | turn_right | open | 0.63 0.24 0.07 0.05 0.01 | 0.63 0.24 0.07 0.05 0.01 | 0 -> 0 | lane_left / lane_left / straight | 2.42 / 2.42 | 0.71 |
| 6.0 | doplan_000013 | stop | open | 0.73 0.22 0.04 0.02 0.00 | 0.70 0.21 0.03 0.02 0.04 | 0 -> 0 | straight / straight / straight | 2.37 / 2.37 | 0.79 |
| 6.0 | doplan_001238 | lane_left | open | 0.73 0.22 0.04 0.02 0.00 | 0.28 0.08 0.63 0.01 0.00 | 0 -> 2 | straight / lane_left / straight | 2.37 / 4.56 | 0.79 |
| 6.0 | doplan_003808 | turn_right | open | 0.73 0.22 0.04 0.02 0.00 | 0.73 0.22 0.04 0.02 0.00 | 0 -> 0 | straight / straight / straight | 2.37 / 2.37 | 0.79 |
| 6.0 | doplan_004144 | lane_right | open | 0.73 0.22 0.04 0.02 0.00 | 0.73 0.22 0.04 0.02 0.00 | 0 -> 0 | straight / straight / straight | 2.37 / 2.37 | 0.79 |
| 6.0 | doplan_006457 | turn_right | open | 0.73 0.22 0.04 0.02 0.00 | 0.73 0.22 0.04 0.02 0.00 | 0 -> 0 | straight / straight / straight | 2.37 / 2.37 | 0.79 |
| 6.5 | doplan_000013 | stop | open | 0.89 0.07 0.03 0.00 0.00 | 0.87 0.07 0.03 0.00 0.02 | 0 -> 0 | straight / straight / straight | 3.00 / 3.00 | 0.41 |
| 6.5 | doplan_001238 | lane_left | open | 0.89 0.07 0.03 0.00 0.00 | 0.89 0.07 0.03 0.00 0.00 | 0 -> 0 | straight / straight / straight | 3.00 / 3.00 | 0.41 |
| 6.5 | doplan_003808 | turn_right | open | 0.89 0.07 0.03 0.00 0.00 | 0.89 0.07 0.03 0.00 0.00 | 0 -> 0 | straight / straight / straight | 3.00 / 3.00 | 0.41 |
| 6.5 | doplan_004144 | lane_right | open | 0.89 0.07 0.03 0.00 0.00 | 0.89 0.07 0.03 0.00 0.00 | 0 -> 0 | straight / straight / straight | 3.00 / 3.00 | 0.41 |
| 6.5 | doplan_006457 | turn_right | open | 0.89 0.07 0.03 0.00 0.00 | 0.89 0.07 0.03 0.00 0.00 | 0 -> 0 | straight / straight / straight | 3.00 / 3.00 | 0.41 |
| 7.0 | doplan_000013 | stop | open | 0.85 0.11 0.02 0.01 0.00 | 0.79 0.11 0.02 0.01 0.08 | 0 -> 0 | straight / straight / straight+stop | 3.92 / 3.92 | 0.81 |
| 7.0 | doplan_001238 | lane_left | open | 0.85 0.11 0.02 0.01 0.00 | 0.85 0.11 0.02 0.01 0.00 | 0 -> 0 | straight / straight / straight+stop | 3.92 / 3.92 | 0.81 |
| 7.0 | doplan_003808 | turn_right | open | 0.85 0.11 0.02 0.01 0.00 | 0.85 0.11 0.02 0.01 0.00 | 0 -> 0 | straight / straight / straight+stop | 3.92 / 3.92 | 0.81 |
| 7.0 | doplan_004144 | lane_right | open | 0.85 0.11 0.02 0.01 0.00 | 0.85 0.11 0.02 0.01 0.00 | 0 -> 0 | straight / straight / straight+stop | 3.92 / 3.92 | 0.81 |
| 7.0 | doplan_006457 | turn_right | open | 0.85 0.11 0.02 0.01 0.00 | 0.85 0.11 0.02 0.01 0.00 | 0 -> 0 | straight / straight / straight+stop | 3.92 / 3.92 | 0.81 |
| 7.5 | doplan_000013 | stop | closed | 0.87 0.11 0.01 0.01 0.00 | 0.87 0.11 0.01 0.01 0.00 | 0 -> 0 | straight / straight / straight+stop | 4.86 / 4.86 | 1.20 |
| 7.5 | doplan_001238 | lane_left | closed | 0.87 0.11 0.01 0.01 0.00 | 0.87 0.11 0.01 0.01 0.00 | 0 -> 0 | straight / straight / straight+stop | 4.86 / 4.86 | 1.20 |
| 7.5 | doplan_003808 | turn_right | closed | 0.87 0.11 0.01 0.01 0.00 | 0.87 0.11 0.01 0.01 0.00 | 0 -> 0 | straight / straight / straight+stop | 4.86 / 4.86 | 1.20 |
| 7.5 | doplan_004144 | lane_right | closed | 0.87 0.11 0.01 0.01 0.00 | 0.87 0.11 0.01 0.01 0.00 | 0 -> 0 | straight / straight / straight+stop | 4.86 / 4.86 | 1.20 |
| 7.5 | doplan_006457 | turn_right | closed | 0.87 0.11 0.01 0.01 0.00 | 0.87 0.11 0.01 0.01 0.00 | 0 -> 0 | straight / straight / straight+stop | 4.86 / 4.86 | 1.20 |
| 8.0 | doplan_000013 | stop | open | 0.73 0.20 0.04 0.03 0.01 | 0.62 0.17 0.03 0.02 0.16 | 0 -> 0 | straight / straight / straight+stop | 6.18 / 6.18 | 3.33 |
| 8.0 | doplan_001238 | lane_left | open | 0.73 0.20 0.04 0.03 0.01 | 0.73 0.20 0.04 0.03 0.01 | 0 -> 0 | straight / straight / straight+stop | 6.18 / 6.18 | 3.33 |
| 8.0 | doplan_003808 | turn_right | open | 0.73 0.20 0.04 0.03 0.01 | 0.73 0.20 0.04 0.03 0.01 | 0 -> 0 | straight / straight / straight+stop | 6.18 / 6.18 | 3.33 |
| 8.0 | doplan_004144 | lane_right | open | 0.73 0.20 0.04 0.03 0.01 | 0.73 0.20 0.04 0.03 0.01 | 0 -> 0 | straight / straight / straight+stop | 6.18 / 6.18 | 3.33 |
| 8.0 | doplan_006457 | turn_right | open | 0.73 0.20 0.04 0.03 0.01 | 0.73 0.20 0.04 0.03 0.01 | 0 -> 0 | straight / straight / straight+stop | 6.18 / 6.18 | 3.33 |
| 8.5 | doplan_000013 | stop | open | 0.50 0.22 0.13 0.08 0.07 | 0.12 0.05 0.03 0.43 0.37 | 0 -> 3 | straight / straight+stop / straight | 5.74 / 3.54 | 1.79 |
| 8.5 | doplan_001238 | lane_left | open | 0.50 0.22 0.13 0.08 0.07 | 0.50 0.22 0.13 0.08 0.07 | 0 -> 0 | straight / straight / straight | 5.74 / 5.74 | 1.79 |
| 8.5 | doplan_003808 | turn_right | open | 0.50 0.22 0.13 0.08 0.07 | 0.50 0.22 0.13 0.08 0.07 | 0 -> 0 | straight / straight / straight | 5.74 / 5.74 | 1.79 |
| 8.5 | doplan_004144 | lane_right | open | 0.50 0.22 0.13 0.08 0.07 | 0.50 0.22 0.13 0.08 0.07 | 0 -> 0 | straight / straight / straight | 5.74 / 5.74 | 1.79 |
| 8.5 | doplan_006457 | turn_right | open | 0.50 0.22 0.13 0.08 0.07 | 0.50 0.22 0.13 0.08 0.07 | 0 -> 0 | straight / straight / straight | 5.74 / 5.74 | 1.79 |
| 9.0 | doplan_000013 | stop | open | 0.28 0.23 0.20 0.18 0.11 | 0.44 0.36 0.02 0.01 0.17 | 0 -> 0 | straight+stop / straight+stop / straight | 5.54 / 5.54 | 2.00 |
| 9.0 | doplan_001238 | lane_left | open | 0.28 0.23 0.20 0.18 0.11 | 0.28 0.23 0.20 0.18 0.11 | 0 -> 0 | straight+stop / straight+stop / straight | 5.54 / 5.54 | 2.00 |
| 9.0 | doplan_003808 | turn_right | open | 0.28 0.23 0.20 0.18 0.11 | 0.28 0.23 0.20 0.18 0.11 | 0 -> 0 | straight+stop / straight+stop / straight | 5.54 / 5.54 | 2.00 |
| 9.0 | doplan_004144 | lane_right | open | 0.28 0.23 0.20 0.18 0.11 | 0.28 0.23 0.20 0.18 0.11 | 0 -> 0 | straight+stop / straight+stop / straight | 5.54 / 5.54 | 2.00 |
| 9.0 | doplan_006457 | turn_right | open | 0.28 0.23 0.20 0.18 0.11 | 0.28 0.23 0.20 0.18 0.11 | 0 -> 0 | straight+stop / straight+stop / straight | 5.54 / 5.54 | 2.00 |
| 9.5 | doplan_000013 | stop | open | 0.26 0.19 0.19 0.19 0.17 | 0.05 0.84 0.04 0.04 0.03 | 0 -> 1 | straight / straight+stop / straight | 5.37 / 0.46 | 0.46 |
| 9.5 | doplan_001238 | lane_left | open | 0.26 0.19 0.19 0.19 0.17 | 0.26 0.19 0.19 0.19 0.17 | 0 -> 0 | straight / straight / straight | 5.37 / 5.37 | 0.46 |
| 9.5 | doplan_003808 | turn_right | open | 0.26 0.19 0.19 0.19 0.17 | 0.26 0.19 0.19 0.19 0.17 | 0 -> 0 | straight / straight / straight | 5.37 / 5.37 | 0.46 |
| 9.5 | doplan_004144 | lane_right | open | 0.26 0.19 0.19 0.19 0.17 | 0.26 0.19 0.19 0.19 0.17 | 0 -> 0 | straight / straight / straight | 5.37 / 5.37 | 0.46 |
| 9.5 | doplan_006457 | turn_right | open | 0.26 0.19 0.19 0.19 0.17 | 0.26 0.19 0.19 0.19 0.17 | 0 -> 0 | straight / straight / straight | 5.37 / 5.37 | 0.46 |
| 10.0 | doplan_000013 | stop | open | 0.35 0.29 0.27 0.05 0.04 | 0.21 0.17 0.16 0.03 0.44 | 0 -> 4 | straight / straight+stop / straight | 4.81 / 1.44 | 1.44 |
| 10.0 | doplan_001238 | lane_left | open | 0.35 0.29 0.27 0.05 0.04 | 0.35 0.29 0.27 0.05 0.04 | 0 -> 0 | straight / straight / straight | 4.81 / 4.81 | 1.44 |
| 10.0 | doplan_003808 | turn_right | open | 0.35 0.29 0.27 0.05 0.04 | 0.35 0.29 0.27 0.05 0.04 | 0 -> 0 | straight / straight / straight | 4.81 / 4.81 | 1.44 |
| 10.0 | doplan_004144 | lane_right | open | 0.35 0.29 0.27 0.05 0.04 | 0.35 0.29 0.27 0.05 0.04 | 0 -> 0 | straight / straight / straight | 4.81 / 4.81 | 1.44 |
| 10.0 | doplan_005651 | lane_left | open | 0.35 0.29 0.27 0.05 0.04 | 0.35 0.29 0.27 0.05 0.04 | 0 -> 0 | straight / straight / straight | 4.81 / 4.81 | 1.44 |
| 10.0 | doplan_006457 | turn_right | open | 0.35 0.29 0.27 0.05 0.04 | 0.35 0.29 0.27 0.05 0.04 | 0 -> 0 | straight / straight / straight | 4.81 / 4.81 | 1.44 |

## Plots

![scene7_t000.0s_doplan_000013](scene7_t000.0s_doplan_000013.png)
![scene7_t000.0s_doplan_003808](scene7_t000.0s_doplan_003808.png)
![scene7_t000.0s_doplan_004144](scene7_t000.0s_doplan_004144.png)
![scene7_t000.0s_doplan_006457](scene7_t000.0s_doplan_006457.png)
![scene7_t002.0s_doplan_000013](scene7_t002.0s_doplan_000013.png)
![scene7_t002.0s_doplan_003808](scene7_t002.0s_doplan_003808.png)
![scene7_t002.0s_doplan_004144](scene7_t002.0s_doplan_004144.png)
![scene7_t002.0s_doplan_006457](scene7_t002.0s_doplan_006457.png)
![scene7_t004.0s_doplan_000013](scene7_t004.0s_doplan_000013.png)
![scene7_t004.0s_doplan_001238](scene7_t004.0s_doplan_001238.png)
![scene7_t004.0s_doplan_003808](scene7_t004.0s_doplan_003808.png)
![scene7_t004.0s_doplan_004144](scene7_t004.0s_doplan_004144.png)
![scene7_t004.0s_doplan_006457](scene7_t004.0s_doplan_006457.png)
![scene7_t006.0s_doplan_000013](scene7_t006.0s_doplan_000013.png)
![scene7_t006.0s_doplan_001238](scene7_t006.0s_doplan_001238.png)
![scene7_t006.0s_doplan_003808](scene7_t006.0s_doplan_003808.png)
![scene7_t006.0s_doplan_004144](scene7_t006.0s_doplan_004144.png)
![scene7_t006.0s_doplan_006457](scene7_t006.0s_doplan_006457.png)
![scene7_t008.0s_doplan_000013](scene7_t008.0s_doplan_000013.png)
![scene7_t008.0s_doplan_001238](scene7_t008.0s_doplan_001238.png)
![scene7_t008.0s_doplan_003808](scene7_t008.0s_doplan_003808.png)
![scene7_t008.0s_doplan_004144](scene7_t008.0s_doplan_004144.png)
![scene7_t008.0s_doplan_006457](scene7_t008.0s_doplan_006457.png)
![scene7_t010.0s_doplan_000013](scene7_t010.0s_doplan_000013.png)
![scene7_t010.0s_doplan_001238](scene7_t010.0s_doplan_001238.png)
![scene7_t010.0s_doplan_003808](scene7_t010.0s_doplan_003808.png)
![scene7_t010.0s_doplan_004144](scene7_t010.0s_doplan_004144.png)
![scene7_t010.0s_doplan_005651](scene7_t010.0s_doplan_005651.png)
![scene7_t010.0s_doplan_006457](scene7_t010.0s_doplan_006457.png)
