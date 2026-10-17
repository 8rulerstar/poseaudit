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

against val2017: straight (labelled 150° or more) and bent arms in the bins of limb length of validation/error_budget.out.md, off by 15° or more. Label noise alone (SE over the demo's arms), then with a model's own independent point noise of 0.5 and 1 times the label spread (sigma*s/2), beside what yolo11x-pose reads on val2017's left elbows

| arms | segments | demo arms | labels alone, sigma*s/2 | model noise 0.5, 1 | labels alone, sigma*s/sqrt(2) | yolo11x read, val2017 |
|---|---|---|---|---|---|---|
| straight | all | 106 | 25% (SE 1.0) | 28%, 36% | 39% | 14% |
| straight | under 16 px | 20 | 29% (SE 2.7) | 36%, 43% | 41% | 20% |
| straight | 16-24 px | 17 | 23% (SE 2.8) | 25%, 32% | 35% | 16% |
| straight | 24-35 px | 18 | 28% (SE 2.2) | 32%, 38% | 40% | 16% |
| straight | 35-54 px | 24 | 22% (SE 2.1) | 25%, 32% | 36% | 9% |
| straight | 54+ px | 27 | 25% (SE 1.9) | 26%, 35% | 39% | 10% |
| bent | all | 216 | 26% (SE 1.1) | 31%, 41% | 41% | 22% |
| bent | under 19 px | 38 | 36% (SE 2.3) | 42%, 51% | 51% | 36% |
| bent | 19-28 px | 42 | 29% (SE 2.3) | 34%, 44% | 44% | 25% |
| bent | 28-41 px | 48 | 32% (SE 2.6) | 35%, 46% | 46% | 20% |
| bent | 41-66 px | 40 | 20% (SE 2.1) | 24%, 35% | 34% | 17% |
| bent | 66+ px | 48 | 16% (SE 1.9) | 20%, 30% | 30% | 14% |

honest models on the same arms, 30 runs each (spreads in units of sigma*s per axis)

| label spread | model spread | mean gap | p(slope <= jitter) <= 0.05 | Deming, ratio 1 |
|---|---|---|---|---|
| 0.0 | 0.71 | +0.002 | 0% | 1.018 |
| 0.5 | 0.5 | -0.021 | 23% | 0.998 |
| 0.71 | 0.71 | -0.040 | 17% | 0.998 |
| 0.5 | 1.0 | -0.020 | 7% | 1.035 |
