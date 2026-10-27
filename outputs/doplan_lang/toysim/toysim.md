# Toy simulator: outcome of the selected trajectories

Pure pursuit on a kinematic bicycle (dt 0.1 s, |a| <= 4.0 m/s^2, |kappa| <= 0.2 1/m), agents replayed from the log (not reactive: collisions after the ego leaves the recorded path are pessimistic). Completion: the executed path performs the instructed manoeuvre by the lane graph (stop: final speed < 1.0 m/s); undefined for "other". Off-lane: any step outside every lane polygon and more than 2.0 m from every lane centreline (riding a lane edge is not off-lane). The recorded execution is the reference: the recorded future through the same tracker.

| subset | execution | n | completion (n) | collision | off-lane | tracking RMS (m) | lat accel p95 (m/s^2) |
|---|---|---|---|---|---|---|---|
| gated | mtr_top1 | 63621 | 0.188 (57211) | 0.092 | 0.053 | 0.235 | 2.156 |
| gated | reranked_top1 | 63621 | 0.344 (58868) | 0.090 | 0.051 | 0.237 | 2.243 |
| gated | recorded | 40341 | 0.222 (36840) | 0.001 | 0.004 | 0.115 | 1.891 |
| all | mtr_top1 | 77127 | 0.187 (69304) | 0.095 | 0.059 | 0.219 | 2.032 |
| all | reranked_top1 | 77127 | 0.317 (70961) | 0.094 | 0.058 | 0.220 | 2.087 |
| all | recorded | 49448 | 0.220 (45172) | 0.001 | 0.003 | 0.103 | 1.794 |

| subset | metric | reranked - MTR [95% CI] |
|---|---|---|
| gated | completion | 0.137 [0.101, 0.173] |
| gated | collision | -0.001 [-0.006, 0.003] |
| gated | offlane | -0.002 [-0.012, 0.006] |
| all | completion | 0.113 [0.084, 0.146] |
| all | collision | -0.001 [-0.005, 0.003] |
| all | offlane | -0.001 [-0.009, 0.005] |

203702 executions over 77127 pairs; CIs resample moments.
