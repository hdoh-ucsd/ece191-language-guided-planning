# word2vec report

- vocabulary: 616 words, 100 dimensions
- evaluation-log token coverage: 0.993
- cosine(left, right): 0.590 (high is expected in a small corpus; the intent classifier compensates)

## nearest neighbours
- **left**: right (0.59), lef (0.58), at (0.56), ahread (0.53), stop (0.52), itnersection (0.52), a (0.52), correct (0.52), section (0.51), the (0.51)
- **right**: left (0.59), at (0.50), the (0.50), correct (0.49), running (0.46), to (0.44), lefts (0.44), now (0.43), turn (0.43), lane (0.43)
- **stop**: afterwards (0.66), continur (0.62), ahread (0.62), apporaching (0.62), a (0.60), upcomong (0.58), reaching (0.57), upcoming (0.57), breifly (0.56), thre (0.55)
- **straight**: striaght (0.54), procceed (0.51), lion (0.51), leo (0.50), awhile (0.48), afterwards (0.46), hit (0.46), rd (0.45), walkers (0.45), continue (0.45)
- **lane**: futher (0.60), correct (0.55), lanes (0.53), section (0.53), megre (0.51), delivery (0.51), police (0.50), athe (0.50), merge (0.50), ove (0.49)
- **merge**: megre (0.65), futher (0.64), mege (0.62), deliverly (0.61), red (0.61), ove (0.60), delivery (0.59), upcoming (0.59), causing (0.58), furtherest (0.58)
- **light**: becomes (0.55), procceed (0.53), intersectio (0.52), intersection (0.52), interesection (0.50), striaght (0.49), mege (0.49), upcomong (0.49), afterwards (0.48), left (0.48)
- **sign**: ahread (0.62), walkway (0.57), continur (0.56), afterward (0.54), a (0.53), afterwards (0.52), upcomong (0.52), itnersection (0.52), apporaching (0.51), rights (0.51)
- **intersection**: mege (0.61), inersection (0.60), sop (0.60), trafific (0.59), apporaching (0.58), intersectio (0.56), interesection (0.56), thre (0.55), make (0.55), encounter (0.54)
- **exit**: beaver (0.53), pedestiran (0.50), eastport (0.46), rd (0.45), preschool (0.45), be (0.44), southbound (0.43), immediately (0.42), using (0.42), flag (0.42)
