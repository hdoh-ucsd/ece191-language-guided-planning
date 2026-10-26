# intent classifier report

- training clauses: 5108
- 5-fold CV accuracy vs rules: 0.957
- per-class F1: turn_left 0.97, turn_right 0.97, lane_left 0.93, lane_right 0.98, stop 0.98, straight 0.95, other 0.83

## confusion (rows = rule label, cols = predicted)

| | turn_left | turn_right | lane_left | lane_right | stop | straight | other |
|---|---|---|---|---|---|---|---|
| turn_left | 1217 | 4 | 12 | 0 | 2 | 4 | 32 |
| turn_right | 0 | 1053 | 1 | 3 | 3 | 1 | 23 |
| lane_left | 3 | 0 | 270 | 0 | 2 | 4 | 3 |
| lane_right | 0 | 7 | 1 | 590 | 0 | 1 | 3 |
| stop | 1 | 4 | 3 | 1 | 941 | 1 | 15 |
| straight | 1 | 4 | 2 | 2 | 3 | 458 | 10 |
| other | 16 | 9 | 9 | 7 | 12 | 11 | 359 |

## clauses the rules could not label (423)

- classifier argmax counts: other 414, turn_left 5, lane_right 2, straight 1, turn_right 1

- "head toward dry dock plaza park" -> other (1.00)
- "go to tunnel park" -> other (0.99)
- "go to paris las vegas" -> other (0.99)
- "go around the stopped cars" -> other (0.98)
- "bring me to the main entrance of the aria" -> other (0.91)
- "head toward luxor las vegas" -> other (1.00)
- "drive along the water" -> other (1.00)
- "drive towards pier 10 park" -> other (1.00)
- "slow down for backing van" -> other (0.79)
- "yield to oncoming vehicles" -> other (0.96)
- "head to hot metal bridge" -> other (1.00)
- "go towards hampton inn & homewood suites" -> other (0.99)
- "drive into the paris hotel entrance" -> other (0.95)
- "exit luxor las vegas and head east" -> other (0.93)
- "go to the entrance of paris las vegas" -> other (0.98)
- "take me across evelyn moakley bridge" -> other (0.98)
- "let me out in the front of this building" -> other (0.97)
- "go to the entrace of pier 10 park" -> other (1.00)
- "go around the roundabout at veer towers east" -> other (0.82)
- "head toward the eifel tower" -> other (1.00)
- "switch to the left lane right" -> lane_right (0.54)
- "allow pedestrians to cross" -> other (0.97)
- "enter into luxor las vegas and exit through it" -> other (0.89)
- "look for oncoming vehicles and leave the drop off zone" -> other (0.53)
- "go across the bridge" -> other (0.95)
- "go around the white car" -> other (1.00)
- "head toward dry dock plaza park" -> other (1.00)
- "get me to the paris hotel" -> other (1.00)
- "make a left turn lake and make the first right turn that is" -> other (0.53)
- "bring me to the front of the luxor" -> other (0.95)
