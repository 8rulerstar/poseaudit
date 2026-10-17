322 arms of the demo, 300 draws

| per-axis spread | mean angle change | off by 15° or more | gain of a perfect model |
|---|---|---|---|
| sigma*s/2 | 11.2° | 25.9% | 0.931 |
| sigma*s/sqrt(2) | 15.6° | 39.9% | 0.872 |

observed: mean 19.48°, 43.2% off by 15° or more, gain 0.731

by arm size (mean labelled segment), off by 15° or more

| arm segments | arms | observed | labels alone, sigma*s/2 | labels alone, sigma*s/sqrt(2) |
|---|---|---|---|---|
| 0-30 px | 134 | 58% | 30% | 44% |
| 30-60 px | 112 | 37% | 27% | 41% |
| 60+ px | 76 | 26% | 17% | 31% |

honest models on the same arms, 30 runs each (spreads in units of sigma*s per axis)

| label spread | model spread | mean gap | p(slope <= jitter) <= 0.05 | Deming, ratio 1 |
|---|---|---|---|---|
| 0.0 | 0.71 | +0.002 | 0% | 1.018 |
| 0.5 | 0.5 | -0.021 | 23% | 0.998 |
| 0.71 | 0.71 | -0.040 | 17% | 0.998 |
| 0.5 | 1.0 | -0.020 | 7% | 1.035 |
