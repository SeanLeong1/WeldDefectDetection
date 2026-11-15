# Circular crown acquisition study, 6 October 2026

Completed campaign: `raised_acquisition_20261006_v1`. The circular crown, renderer, noise law, CNN architecture and neighbouring-median algorithm are unchanged. The experiment changes acquisition sampling and CNN negative-tile sampling.

Independent backgrounds: {'train': 24, 'val': 24, 'test': 24, 'stress': 12}. Scenes: {'train': 96, 'val': 96, 'test': 120, 'stress': 84}. Every split is disjoint and all prior artifacts in the frozen inventory are unchanged.

## Fresh confirmation results

The confirmation set deliberately balances four dwell ranges from 0.1 to 20 microseconds. These are results on the same fresh images, not a comparison of scores from different test populations. CNN seeds and operating points were selected using validation only.

| Method | Box F1 | 95% background interval | Precision | Recall | Clean rings flagged |
|---|---:|---|---:|---:|---:|
| Frozen 5 Oct CNN | 0.020 | [0.014, 0.032] | 0.010 | 0.875 | 10/24 |
| Coverage CNN | 0.851 | [0.724, 0.972] | 0.829 | 0.875 | 1/24 |
| Hard-negative CNN | 0.923 | [0.859, 0.979] | 0.930 | 0.917 | 0/24 |
| New median | 0.315 | [0.204, 0.470] | 0.259 | 0.403 | 1/24 |
| Frozen 5 Oct median | 0.348 | [0.267, 0.443] | 0.364 | 0.333 | 1/24 |

The median recalibration is not an F1 improvement: 0.348 to 0.315. Recall changes from 0.333 to 0.403, while false regions increase from 42 to 83. Both results remain visible. The comparator is not switched retrospectively using confirmation.

**Remaining limitation: ring localisation.** The primary confirmation uses declared ring coordinates. On the same separate 84-scene robustness population, the selected model scores 0.616 F1 with nominal coordinates and 0.351 with image-estimated coordinates. The acquisition improvement does not establish localisation robustness. Do not interpret the main-confirmation score as an end-to-end unknown-image result.

The GUI default is **hard_negative**, selected before confirmation. This choice is not replaced by whichever model has the highest confirmation score.

All confirmation metrics above include 120 scenes, including hidden controls. Those controls exactly repeat the clean surface signal and are grouped with their background in uncertainty calculations. They are not independent additional negative examples.

| Method | F1 excluding duplicate hidden controls (96 scenes) | Hidden controls flagged |
|---|---:|---:|
| Frozen 5 Oct CNN | 0.024 | 10/24 |
| Coverage CNN | 0.857 | 1/24 |
| Hard-negative CNN | 0.923 | 0/24 |
| New median | 0.333 | 1/24 |
| Frozen 5 Oct median | 0.350 | 1/24 |

## Concentration of remaining errors

| Arm | Highest-FP background | Dwell us | False regions across paired scenes | False regions on clean scene | Missed regions |
|---|---|---:|---:|---:|---:|
| coverage | test-000 | 0.112 | 6 | 1 | 2 |
| coverage | test-002 | 2.32 | 2 | 0 | 2 |
| coverage | test-015 | 14.5 | 2 | 0 | 2 |
| hard_negative | test-000 | 0.112 | 1 | 0 | 2 |
| hard_negative | test-002 | 2.32 | 1 | 0 | 1 |
| hard_negative | test-009 | 0.89 | 1 | 0 | 1 |

These background rankings are descriptive. They must not be used to retune the frozen models or discard difficult confirmation scenes.

## Surface-condition recovery

| Method | Open keyhole-shaped pore recall | Open intrinsic-pore recall | Assumed bulge recall |
|---|---:|---:|---:|
| Frozen 5 Oct CNN | 0.833 | 0.917 | 0.875 |
| Coverage CNN | 0.833 | 0.792 | 1.000 |
| Hard-negative CNN | 0.833 | 0.917 | 1.000 |
| New median | 0.167 | 0.167 | 0.875 |
| Frozen 5 Oct median | 0.167 | 0.083 | 0.750 |

Each positive condition contains 24 scenes. Recall alone ignores false regions; use the overall precision and clean-ring alarm results alongside this table. Bulges remain an assumed surface-displacement case, not evidence of buried-pore visibility.

## Paired comparisons

| Comparison | F1 difference | 95% paired interval | Recall difference | Clean alarm-rate difference |
|---|---:|---|---:|---:|
| hard minus coverage | +0.072 | [+0.014, +0.136] | +0.042 | -0.042 |
| coverage minus frozen | +0.832 | [+0.694, +0.942] | +0.000 | -0.375 |
| hard minus frozen | +0.904 | [+0.832, +0.957] | +0.042 | -0.417 |

For the validation-selected fitted models, the paired bootstrap interval favours the hard-negative model on this synthetic population. This is not a guarantee of improvement across initialisations.

Intervals use 2,000 paired replicates, resampling independent backgrounds within dwell bins. Only six confirmation backgrounds are available per bin. These intervals condition on the fitted models and observed synthetic backgrounds, and do not include uncertainty about real-world domain shift. Three trained seeds show additional optimisation variability.

If a method flags 0/24 clean rings, its empirical bootstrap alarm interval is degenerate at zero. Even under an additional independent, identically distributed Bernoulli assumption, the one-sided 95% upper bound would be about 11.7%, not zero; the intentionally stratified design is not proof of a deployment alarm rate.

Dwell-bin summaries are stratified diagnostics, not a within-background causal dose-response. The inherited index-based layouts do not form a full factorial of seam position, crown variation and defect size. The matched training-arm comparison uses exactly the same scenes in both arms.

![Dwell-stratified results](figures/raised_acquisition_20261006/dwell_results.png)

## All seeds and validation choices

| Arm | Seed | Validation F1 | Confirmation F1 | Threshold | Area | Closing | Selected within arm |
|---|---:|---:|---:|---:|---:|---:|---|
| coverage | 100606001 | 0.745 | 0.733 | 0.8 | 15 | 3 | False |
| coverage | 100606002 | 0.875 | 0.851 | 0.9 | 15 | 3 | True |
| coverage | 100606003 | 0.810 | 0.785 | 0.5 | 15 | 3 | False |
| hard_negative | 100606001 | 0.792 | 0.729 | 0.65 | 15 | 3 | False |
| hard_negative | 100606002 | 0.932 | 0.923 | 0.35 | 15 | 3 | True |
| hard_negative | 100606003 | 0.788 | 0.841 | 0.97 | 15 | 3 | False |

![Seed variability](figures/raised_acquisition_20261006/seed_variability.png)

## Multiple defects and ring localisation

Stress includes clean rings, separated, near and overlapping pairs, triples, small openings and hidden controls. Nominal and image-estimated coordinates use all 84 scenes. The combined 0.25 mm x/y/radius error applies only to clean and near-pair scenes, so its matched nominal subset is listed separately.

| Method | Coordinate condition | Box F1 | Region recall | Individual-defect recall | Clean rings flagged |
|---|---|---:|---:|---:|---:|
| coverage | nominal | 0.635 | 0.620 | 0.558 | 1/12 |
| coverage | image_estimated | 0.515 | 0.546 | 0.492 | 3/12 |
| coverage | combined | 0.773 | 0.708 | 0.708 | 0/12 |
| coverage | nominal_combined_subset | 0.766 | 0.750 | not aggregated | 1/12 |
| hard_negative | nominal | 0.616 | 0.639 | 0.583 | 2/12 |
| hard_negative | image_estimated | 0.351 | 0.667 | 0.608 | 5/12 |
| hard_negative | combined | 0.366 | 0.708 | 0.708 | 8/12 |
| hard_negative | nominal_combined_subset | 0.720 | 0.750 | not aggregated | 2/12 |
| median | nominal | 0.119 | 0.111 | 0.100 | 1/12 |
| median | image_estimated | 0.017 | 0.046 | 0.042 | 6/12 |
| median | combined | 0.008 | 0.083 | 0.083 | 12/12 |
| median | nominal_combined_subset | 0.091 | 0.083 | not aggregated | 1/12 |

Overlapping defects may merge into one visible footprint. Region recall and individual-defect recall therefore answer different questions; one merged prediction cannot be counted as several correctly localised defects.

## What changed, chronologically

1. The original 0.1–20 microsecond acquisition range is divided into four strata before rendering. Each independent background gets a declared dwell value. The same unchanged renderer creates related clean and defective scenes. This improves coverage without weakening noise or increasing pore strength.
2. Train-only quantiles define the same clipped square-root input transform. The median calibration is fitted only to clean training images. Validation and confirmation do not alter either transform.
3. Both CNN arms retain the existing 262,777-parameter TinyDefectSegNet, four radial tiles, weighted binary cross-entropy plus Dice loss, and 20 epochs with 1,200 samples per epoch. Their positive samples and initial weights match by seed.
4. The coverage arm samples all negative tiles uniformly. After epochs 5, 10 and 15, the second arm scores a fixed sample of clean training patches. Patches with high predicted defect probability are difficult negatives. Keeping 16 from each background prevents one noisy ring from taking the entire bank. Subsequent batches contain 600 positive, 300 uniform negative and 300 mined negative tiles. Mining adds forward-pass compute; it does not add optimizer updates.
5. Validation selects the final-epoch seed and threshold/area/closing settings. Both arms and the median are frozen before fresh confirmation is generated. Test results cannot feed back into this experiment.
6. Evaluation scores complete rings with periodic connectivity, one-to-one box matching at IoU 0.5, and background-level uncertainty. The GUI uses the same input contract and frozen choices, never truth masks.

![Actual mined clean training patches](figures/raised_acquisition_20261006/mined_clean_examples.png)

For the unchanged CNN layers, geometry-to-image calculations and median operations, see the [chronological method guide](RAISED_CROWN_METHODS_20261005.md). The [prospective protocol](RAISED_ACQUISITION_PROTOCOL_20261006.md) records the comparison before fitting.

## Interpretation and next steps

This tests a training-process improvement inside the existing circular synthetic thesis scope. Comparison with the old frozen CNN includes changed training draws, seeds and a larger validation set (24 rather than 12 backgrounds), so it is not a pure causal estimate of dwell coverage alone. The matched two-arm comparison specifically tests the mining schedule on this dataset.

The next controlled experiment should address localisation-aware training and validation: perturb native-image extraction coordinates, preserve the physical surface and fixed truth coordinates, and compare nominal-only training with coordinate-perturbed training. Select settings on development/validation and evaluate once on fresh backgrounds. The now-exposed stress scenes can motivate the design but cannot remain its untouched confirmation.

Also replicate across more independent low-dwell clean backgrounds and initialisations. The median needs a separate development study of reference contamination and acquisition-dependent residual calibration; its refit here did not improve F1. Do not retrospectively switch this campaign's comparator or thresholds. Real-image validation remains necessary for a physical inspection claim. Unfamiliar weld profiles and electron-transport simulation remain outside this increment.

GUI checks establish that generation, inference and export work and that labels are not used. They do not require every generated clean or hidden-control example to have zero false alarms. Zero observed alarms in 24 confirmation backgrounds is not a guarantee for subsequent examples.

## Run and inspect

Use `Launch Weld Vision.cmd` or `launch_weld_studio.py`. The new acquisition tab exposes the frozen CNN arms and a dwell control, while the 5 October crown and older recessed workflows remain labelled historical. Generated examples are saved separately from research datasets.

```powershell
.venv\Scripts\python.exe WeldDefectDetection/run_raised_acquisition_20261006.py all
```

Evidence: `DataSimulation/data/output/raised_acquisition_20261006_v1/` contains the protocol, source hashes, training histories and mining coordinates, frozen models, scene manifests, predictions, robustness results and audit. Historical datasets, models and release pointers remain unchanged.

Synthetic evidence only. The raised crown is prescribed geometry. Pore-induced bulges are an explicit displacement hypothesis, not a validated welding-fluid model. No researched SEM/BSE penetration number was implemented.
