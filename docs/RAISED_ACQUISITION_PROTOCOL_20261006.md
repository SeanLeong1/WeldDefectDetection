# Circular crown: acquisition coverage and clean-negative experiment

This protocol precedes new model fitting and confirmation. Campaign: `raised_acquisition_20261006_v1`. The previous 5 October data, model selections, results and release remain frozen.

The prior study exposed a concrete acquisition gap. Its clean training backgrounds had dwell times from 1.56 to 19.90 microseconds, although the existing simulator permits 0.1 to 20. A validation image at 0.171 microseconds caused most validation false detections. Rerendering that background with only dwell changed reduced those detections, but that single diagnostic did not establish a general cure.

## Execution order

1. Freeze the new generation protocol and inventory historical artifacts. Keep the same circular 3 mm crown, fusion geometry, pore law, physical pitch, detector model, noise law and CNN architecture. No new electron penetration cutoff is implemented.
2. Generate 24 fresh training and 24 fresh validation backgrounds. Assign six backgrounds to each dwell interval: 0.1–0.5, 0.5–2, 2–8 and 8–20 microseconds. Stratify log dwell within each interval. Include the two global endpoints in training. All four related scenes, clean, open keyhole-shaped pore, open intrinsic pore and assumed bulge, stay with their background.
3. Fit train-only normalization. Compare two CNN arms on exactly the same data, with three common initialization seeds, 20 epochs and 1,200 patches per epoch. Architecture, optimizer, loss, positive samples and update count remain fixed. Both begin with 600 positive and 600 uniformly sampled negative tiles per epoch.
4. In the hard-negative arm, after epochs 5, 10 and 15, score 168 clean training tiles per background with the current CNN. Rank by the mean of the highest 64 pixel probabilities and retain 16 per background. Replace 300 of the 600 negative draws with draws from that bank. Retain 300 uniform negatives. This limits domination by a single background. Mining never reads validation, confirmation or stress data. It adds scoring compute, not optimizer updates.
5. Refit the same neighbouring-median comparator from clean training images. Select final-epoch CNN seeds and operating points using validation only, with the unchanged threshold/area/closing grids. Freeze both arms, the median and the validation-selected GUI default.
6. Only then generate 24 fresh confirmation backgrounds with the same four dwell strata. Include the four surface classes and an enclosed/no-surface-response control. Generate 12 additional stress backgrounds with the existing multiple-defect and small-pore layouts. Use the same native-coordinate extraction and image-only ring estimator. The combined localisation perturbation is 0.25 mm in each of x, y and radius, on matched clean/close-pair scenes.
7. Evaluate every CNN seed, selected arms, new median and frozen 5 October selected methods. Report full-ring box F1, precision, recall, clean-ring alarms, individual-defect recall, dwell-bin results and paired background-bootstrap differences. Preserve both successful and unsuccessful results. Integrate the completed experiment with the GUI and documentation only after integrity and functional checks.

## What this comparison can establish

The matched hard-negative versus coverage comparison isolates the effect of the specified mining schedule conditional on the dataset and validation procedure. It retains three seeds to show training variability. The comparison with the frozen 5 October model measures improvement on new synthetic data; it does not isolate dwell coverage alone because the random training backgrounds and seeds also change. A full paired rendering/training factorial would be needed for that stronger causal claim.

Confirmation is an intentionally balanced acquisition challenge, not an estimate of natural deployment frequency. Low dwell is deliberately represented more often than in the old random draw. Dwell changes both signal/noise and raster timing in the existing acquisition model. No noise setting is weakened to improve F1.

Resample independent backgrounds within dwell strata, preserving related scenes and matching methods in 2,000 bootstrap replicates. With six confirmation backgrounds per stratum, estimates remain exploratory. Zero observed clean alarms cannot prove a zero population alarm rate. The hidden control duplicates its clean surface signal by construction; report its behaviour separately and also provide metrics excluding that duplicate. Neither hidden-control performance nor these synthetic scores establish real SEM information depth or real-image transfer.

The primary comparison is hard-negative minus coverage box F1. False alarms and recall must be read alongside F1: suppressing every detection is not an improvement. There is no test-driven threshold adjustment, seed replacement or subsequent retraining within this campaign.

## Reproduce

```powershell
.venv\Scripts\python.exe WeldDefectDetection/run_raised_acquisition_20261006.py all
```

The runner resumes checksum-verified scenes and checkpoints. CPU workers generate disjoint backgrounds. Historical evidence and fresh confirmation must never be silently recycled into development data.
