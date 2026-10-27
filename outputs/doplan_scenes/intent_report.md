# intent classifier report

- training clauses: 5190
- 5-fold CV accuracy vs rules: 0.951
- per-class F1: turn_left 0.97, turn_right 0.97, lane_left 0.92, lane_right 0.97, stop 0.98, straight 0.95, other 0.81

## confusion (rows = rule label, cols = predicted)

| | turn_left | turn_right | lane_left | lane_right | stop | straight | other |
|---|---|---|---|---|---|---|---|
| turn_left | 1272 | 5 | 13 | 0 | 4 | 6 | 37 |
| turn_right | 3 | 1053 | 1 | 4 | 2 | 2 | 30 |
| lane_left | 6 | 0 | 260 | 0 | 1 | 4 | 9 |
| lane_right | 0 | 6 | 2 | 586 | 1 | 1 | 6 |
| stop | 1 | 5 | 2 | 2 | 942 | 2 | 15 |
| straight | 3 | 3 | 1 | 2 | 1 | 466 | 9 |
| other | 13 | 10 | 8 | 8 | 10 | 17 | 356 |

## clauses the rules could not label (422)

- classifier argmax counts: other 406, turn_left 8, lane_left 5, turn_right 2, straight 1

- "bring me to the front of the paris hotel" -> other (0.99)
- "head toward dry dock plaza park" -> other (1.00)
- "go toward eastport park" -> other (0.98)
- "head toward the exit" -> other (0.99)
- "take me towards citgo" -> other (0.99)
- "head toward dry dock plaza park" -> other (1.00)
- "go toward the exit" -> other (0.95)
- "head toward dry dock plaza park" -> other (1.00)
- "head toward harbour plaza" -> other (1.00)
- "go into paris las vegas roundabout" -> other (0.98)
- "head back onto the main road toward the exit" -> other (0.94)
- "the taxi passes you go into the main road and head toward the exit" -> other (0.99)
- "let me out in front of the luxor" -> other (0.95)
- "make the same turn as the vehicle in front of you" -> other (0.49)
- "allow the truck to back up and clear the road" -> other (0.83)
- "make a right turn at the stop sign and hten switch to the second left lane" -> other (0.68)
- "drive around the yellow trucks" -> other (0.50)
- "temporarily park at the curb" -> other (0.96)
- "get me to the paris hotel" -> other (1.00)
- "head over hot metal bridge towards tunnel park" -> other (0.94)
- "head toward eastport park" -> other (1.00)
- "folllow the white car in front of you" -> other (0.72)
- "drive along the water" -> other (1.00)
- "slow down for the pedestrian crossing the street" -> other (0.83)
- "head toward the institute of contemporary art" -> other (1.00)
- "head to the luxor entrance" -> other (0.99)
- "stay on the left side of the road to avoid the parked vehicles on the right and make a left turn" -> turn_left (0.97)
- "go to the entrance of paris las vegas" -> other (0.99)
- "head to tunnel park" -> other (0.88)
- "head toward the renaissance boston seaport district" -> other (1.00)
