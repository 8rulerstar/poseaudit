600 runs per case; 200 resamples and 200 rebuilds per run

| case | p(slope <= jitter) <= 0.05 | interval wholly below 0 | setup |
|---|---|---|---|
| isotropic | 4.3% | 2.2% | no tendency; every point scatters 8% of the arm |
| across forearm | 5.7% | 2.8% | no tendency; the wrist scatters 20% across the forearm |
| straight noisier | 2.3% | 4.3% | no tendency; three times the noise above 150 degrees |
| bend bisector | 4.3% | 3.0% | no tendency; the elbow scatters along the bend's bisector |
| random failures | 4.7% | 3.5% | no tendency; 10% of wrists thrown in a random direction |
| label noise | 37.2% | 26.5% | no tendency; the labels scatter as much as the model |
| isotropic, 40 arms | 6.3% | 3.2% | as isotropic, with 40 arms |
| straight noisier, 40 arms | 8.0% | 7.5% | as straight noisier, with 40 arms |
| squash 7% | 100.0% | 100.0% | reads angles 7% closer to 110 degrees, plus isotropic noise |
| squash 15% | 100.0% | 100.0% | reads angles 15% closer to 110 degrees, plus isotropic noise |
