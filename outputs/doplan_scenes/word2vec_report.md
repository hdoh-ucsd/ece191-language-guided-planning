# word2vec report

- vocabulary: 621 words, 100 dimensions
- evaluation-log token coverage: 1.000
- cosine(left, right): 0.616 (high is expected in a small corpus; the intent classifier compensates)

## nearest neighbours
- **left**: right (0.62), the (0.54), at (0.52), lef (0.52), stop (0.51), ahread (0.51), turn (0.51), itnersection (0.50), shiftly (0.50), intersectio (0.50)
- **right**: left (0.62), the (0.55), correct (0.51), running (0.47), then (0.46), a (0.46), at (0.46), to (0.45), lane (0.45), turn (0.44)
- **stop**: afterwards (0.66), ahread (0.63), apporaching (0.63), upcomong (0.62), continur (0.62), breifly (0.59), a (0.59), broken (0.58), thre (0.58), at (0.57)
- **straight**: striaght (0.54), awhile (0.53), leo (0.53), lion (0.53), procceed (0.51), without (0.48), afterwards (0.48), hit (0.47), miles (0.47), walkers (0.45)
- **lane**: futher (0.59), section (0.54), merge (0.53), correct (0.51), delivery (0.51), lanes (0.51), athe (0.51), furtherest (0.50), ove (0.50), megre (0.49)
- **merge**: megre (0.66), mege (0.65), red (0.64), futher (0.63), furtherest (0.62), causing (0.61), deliverly (0.60), ove (0.60), speed (0.59), correct (0.58)
- **light**: becomes (0.57), upcomong (0.54), procceed (0.52), thre (0.51), lights (0.50), afterwards (0.50), reaching (0.50), striaght (0.49), intersectio (0.49), mege (0.49)
- **sign**: ahread (0.59), upcomong (0.55), afterward (0.54), walkway (0.54), continur (0.53), stop (0.52), lft (0.52), afterwards (0.51), itnersection (0.51), broken (0.50)
- **intersection**: inersection (0.60), sop (0.59), mege (0.58), trafific (0.57), immediate (0.57), apporaching (0.54), stop (0.54), make (0.53), thre (0.53), intersectio (0.53)
- **exit**: beaver (0.51), pedestiran (0.49), rd (0.47), eastport (0.46), without (0.45), flag (0.45), dont (0.44), southbound (0.43), using (0.42), upto (0.42)
