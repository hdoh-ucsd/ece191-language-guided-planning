# Toy simulator: outcome of the selected trajectories

Pure pursuit on a kinematic bicycle (dt 0.1 s, |a| <= 4.0 m/s^2, |kappa| <= 0.2 1/m), agents replayed from the log (not reactive: collisions after the ego leaves the recorded path are pessimistic). Completion: the executed path performs the instructed manoeuvre by the lane graph (stop: final speed < 1.0 m/s); undefined for "other". Off-lane: any step outside every lane polygon and more than 2.0 m from every lane centreline (riding a lane edge is not off-lane). The recorded execution is the reference: the recorded future through the same tracker.

| subset | execution | n | completion (n) | collision | off-lane | tracking RMS (m) | lat accel p95 (m/s^2) |
|---|---|---|---|---|---|---|---|
| gated | mtr_top1 | 180 | 0.122 (147) | 0.000 | 0.000 | 0.120 | 2.146 |
| gated | reranked_top1 | 180 | 0.143 (147) | 0.011 | 0.006 | 0.149 | 2.149 |
| gated | recorded | 117 | 0.170 (94) | 0.000 | 0.000 | 0.064 | 2.407 |
| all | mtr_top1 | 197 | 0.112 (161) | 0.000 | 0.000 | 0.124 | 2.146 |
| all | reranked_top1 | 197 | 0.130 (161) | 0.010 | 0.005 | 0.150 | 2.146 |
| all | recorded | 128 | 0.155 (103) | 0.000 | 0.000 | 0.061 | 2.407 |

| subset | metric | reranked - MTR [95% CI] |
|---|---|---|
| gated | completion | 0.020 [0.000, 0.049] |
| gated | collision | 0.011 [0.000, 0.029] |
| gated | offlane | 0.006 [0.000, 0.018] |
| all | completion | 0.019 [0.000, 0.043] |
| all | collision | 0.010 [0.000, 0.027] |
| all | offlane | 0.005 [0.000, 0.016] |

522 executions over 197 pairs; CIs resample moments.
