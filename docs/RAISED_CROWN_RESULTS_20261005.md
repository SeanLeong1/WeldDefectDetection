# Raised circular crown: results and limits, 5 October 2026

This is a separate synthetic study. It does not establish measured SEM realism, subsurface-pore sensitivity or manufacturing acceptance. Published electron-depth numbers were researched but not implemented.

## Geometry and signal model

The target crown is 3 mm above the original surface; the solid fusion region reaches 7 mm below it, with widths of 2.8 mm at the original surface and 1.2 mm at 5 mm depth. Crown footprint is provisionally 2.8 mm. Plate thickness is not required for the surface experiment.

The old assumed emission model rejects slopes above 60 degrees. This study uses the supported explicit uncalibrated Lambertian fallback, uniform alloy composition, geometric surface pore response and nonlocal ray visibility. Thus old-model transfer measures a joint distribution change. The paired recessed/raised diagnostic holds these new rendering assumptions fixed.

Open surface pores, prescribed volume-displacement bulges and enclosed pores without surface change are separate scenarios. The displacement identity is a mathematical assumption, not experimentally established weld mechanics.

![Surface, signal and labels](figures/raised_crown_results_20261005/surface_signal_labels.png)

## Data and selection

Independent backgrounds by split: {'train': 24, 'val': 12, 'test': 24, 'stress': 12}. Scene counts: {'train': 96, 'val': 48, 'test': 120, 'stress': 84}. Related conditions stay in the same split. Test and stress scenes were generated after model freeze.

The CNN architecture remains TinyDefectSegNet with base width 24, trained for 20 epochs on three seeds. The physical scored band widens to 6.4 mm at 0.05 mm sampling; overlapping 64-pixel tiles cover it without shrinking the weld. Median calibration uses clean training rings. Thresholds, component area and closing are selected on validation only.

## Confirmation

| Model | F1 | Precision | Recall | Clean rings flagged | F1 95% background interval |
|---|---:|---:|---:|---:|---|
| cnn_100505001 | 0.770 | 0.697 | 0.861 | 1/24 | [0.567, 0.938] |
| cnn_100505002 | 0.168 | 0.093 | 0.861 | 1/24 | [0.056, 0.951] |
| cnn_100505003 | 0.190 | 0.106 | 0.903 | 3/24 | [0.075, 0.889] |
| median | 0.290 | 0.346 | 0.250 | 0/24 | [0.220, 0.358] |
| frozen_v4_cnn | 0.064 | 0.075 | 0.056 | 0/24 | [0.015, 0.129] |
| frozen_v4_median | 0.075 | 0.081 | 0.069 | 0/24 | [0.016, 0.138] |

Zero alarms among 24 clean rings does not establish a zero population false-alarm rate. Under an independent identically sampled Bernoulli model, the one-sided 95% upper bound for 0/24 is 1 - 0.05^(1/24), approximately 11.7%. A bootstrap containing only observed zeros is degenerate and cannot express that uncertainty.


Selected CNN: `cnn_100505003`. Selection was made before confirmation; other seeds are replications, not candidates selected using test scores.

Historical transfer uses original weights, normalisation and operating points with an explicit radial tiling adapter. It is not a rerun of the original narrower coverage contract. Hidden controls are counted as negative surface truth and are also reported separately in the machine-readable results.

Clean and hidden/no-response controls share the same simulated surface signal. They are paired conditions, not additional independent negative backgrounds. The primary table includes all five declared conditions. For transparency, the following descriptive view excludes the duplicate hidden-control condition and retains the same frozen models and thresholds.

| Model | Four-condition F1 | Precision | Recall |
|---|---:|---:|---:|
| cnn_100505003 | 0.226 | 0.129 | 0.903 |
| median | 0.290 | 0.346 | 0.250 |

| Model | Surface scenario | True detections | False regions | Misses | Recall |
|---|---|---:|---:|---:|---:|
| cnn_100505003 | open_keyhole | 20 | 114 | 4 | 0.833 |
| cnn_100505003 | open_intrinsic | 23 | 111 | 1 | 0.958 |
| cnn_100505003 | bulge | 22 | 104 | 2 | 0.917 |
| median | open_keyhole | 2 | 14 | 22 | 0.083 |
| median | open_intrinsic | 1 | 10 | 23 | 0.042 |
| median | bulge | 15 | 10 | 9 | 0.625 |

### Validation-selection sensitivity

The selected seed is 100505003, with validation F1 0.134. Replaying the frozen validation inference reproduces the selection metrics exactly. Background `val-011` accounts for 373 of 381 validation false regions. Its clean-image high clipping fraction is 0.00002.

Parameters outside the observed clean-training sample range: ['beam.dwell_time_us']. This is a descriptive association, not proof of a causal mechanism or an out-of-spec acquisition. The selected seed is retained even if another seed scores better on confirmation.

![Validation false alarms](figures/raised_crown_results_20261005/validation_false_alarms.png)


### Controlled dwell diagnostic

Same background, detector, geometry, seed, normalization, model and operating point; only dwell_time_us changed. Original validation pixels were reproduced exactly.

| Condition | Dwell (microseconds) | CNN false regions | Median false regions | Noise RMS |
|---|---:|---:|---:|---:|
| original | 0.1706 | 92 | 0 | 0.048991 |
| training_minimum | 1.5596 | 7 | 0 | 0.032730 |

One validation-background diagnostic within an assumed raster acquisition model. Dwell changes shot/readout scaling and frame/drift timing together. It is not real SEM calibration or a new confirmation score. None; no scene is replaced, no threshold or trained model changed.


## Robustness

| Method / condition | F1 | Recall | Individual-pore recall |
|---|---:|---:|---:|
| cnn / combined | 0.189 | 0.750 | 0.750 |
| cnn / dr_+0.50 | 0.189 | 0.750 | 0.750 |
| cnn / dr_-0.50 | 0.189 | 0.750 | 0.750 |
| cnn / dx_+0.50 | 0.164 | 0.708 | 0.708 |
| cnn / dx_-0.50 | 0.159 | 0.750 | 0.750 |
| cnn / dy_+0.50 | 0.155 | 0.708 | 0.708 |
| cnn / dy_-0.50 | 0.169 | 0.708 | 0.708 |
| cnn / image_estimated | 0.199 | 0.750 | 0.692 |
| cnn / nominal | 0.269 | 0.769 | 0.708 |
| median / combined | 0.008 | 0.042 | 0.042 |
| median / dr_+0.50 | 0.103 | 0.083 | 0.083 |
| median / dr_-0.50 | 0.103 | 0.083 | 0.083 |
| median / dx_+0.50 | 0.000 | 0.000 | 0.000 |
| median / dx_-0.50 | 0.000 | 0.000 | 0.000 |
| median / dy_+0.50 | 0.009 | 0.042 | 0.042 |
| median / dy_-0.50 | 0.019 | 0.083 | 0.083 |
| median / image_estimated | 0.031 | 0.065 | 0.058 |
| median / nominal | 0.108 | 0.083 | 0.075 |

Nominal and image-estimated conditions include all layouts; injected-error sweeps include clean and close pairs only. Compare matched populations.

Predictions are mapped back to fixed native truth. Lost coverage counts as a miss. Instance recall is separate from union-region detection, so one merged region is not automatically credited with finding every overlapping pore. The image-based ring estimator is a bounded image-only baseline, not a validated general localisation system.

### Matched localisation comparison

The following table uses only clean and close-pair scenes for every condition. This avoids comparing injected errors on one population with nominal performance on a different population.

| Method | Condition | F1 | Recall |
|---|---|---:|---:|
| cnn | combined | 0.189 | 0.750 |
| cnn | dr_+0.50 | 0.189 | 0.750 |
| cnn | dr_-0.50 | 0.189 | 0.750 |
| cnn | dx_+0.50 | 0.164 | 0.708 |
| cnn | dx_-0.50 | 0.159 | 0.750 |
| cnn | dy_+0.50 | 0.155 | 0.708 |
| cnn | dy_-0.50 | 0.169 | 0.708 |
| cnn | image_estimated | 0.155 | 0.708 |
| cnn | nominal | 0.222 | 0.750 |
| median | combined | 0.008 | 0.042 |
| median | dr_+0.50 | 0.103 | 0.083 |
| median | dr_-0.50 | 0.103 | 0.083 |
| median | dx_+0.50 | 0.000 | 0.000 |
| median | dx_-0.50 | 0.000 | 0.000 |
| median | dy_+0.50 | 0.009 | 0.042 |
| median | dy_-0.50 | 0.019 | 0.083 |
| median | image_estimated | 0.034 | 0.083 |
| median | nominal | 0.054 | 0.042 |

On 12 independent clean stress backgrounds, the image-only estimator has median/max centre error 0.250/0.707 mm and median/max absolute radius error 0.005/0.014 mm. These values are relative to the known synthetic ring and the estimator's bounded search prior.


### Individual scene types at nominal localisation

| Method | Scene type | F1 | Recall | False regions |
|---|---|---:|---:|---:|
| cnn | clean | n/a | n/a | 59 |
| cnn | hidden | n/a | n/a | 59 |
| cnn | pair_far | 0.423 | 0.917 | 58 |
| cnn | pair_near | 0.350 | 0.750 | 61 |
| cnn | pair_overlap | 0.212 | 0.750 | 64 |
| cnn | small_open | 0.052 | 0.167 | 63 |
| cnn | triple | 0.489 | 0.889 | 63 |
| median | clean | n/a | n/a | 0 |
| median | hidden | n/a | n/a | 0 |
| median | pair_far | 0.108 | 0.083 | 11 |
| median | pair_near | 0.054 | 0.042 | 12 |
| median | pair_overlap | 0.095 | 0.083 | 8 |
| median | small_open | 0.286 | 0.167 | 0 |
| median | triple | 0.103 | 0.083 | 19 |

### Paired profile diagnostic

Paired rendering diagnostic only, not adequately powered transfer confirmation. Held constant: B=0, uniform alloy, acquisition, pore law, sizes, positions and seeds.

| Method | Profile | F1 | Recall |
|---|---|---:|---:|
| cnn | raised | 0.545 | 1.000 |
| cnn | recessed | 0.500 | 0.833 |
| median | raised | 0.500 | 0.500 |
| median | recessed | 0.667 | 0.833 |

### Post-freeze small-pore diagnosis

This diagnostic varies only the component-area filter on the same frozen scores and threshold. It does not change the deployed operating point.

| Minimum component area (pixels) | F1 | Recall |
|---:|---:|---:|
| 1 | 0.042 | 0.167 |
| 3 | 0.045 | 0.167 |
| 15 | 0.052 | 0.167 |

At the production operating point, 5/12 small-pore scenes never reach threshold inside the truth footprint; another 5 have an above-threshold pixel there but still fail the region-matching criterion. This separates weak score response from unsuccessful region localisation. Lowering the area cutoff alone does not resolve those errors.


Sampling sweep evaluates geometric footprints, not a full high-resolution detector/noise rerender. Detailed phase/pitch footprint errors and within-truth score maxima are saved in `small_pore_diagnostic.json`.

### Deterministic small-pore signal sampling

Same physical 2x2 mm field, crown, pore, detector, unit yield modifiers and fixed physical Gaussian blur; compare central 1x1 mm. Isolated deterministic detector-signal diagnostic, not a full noisy raster_v2 rerender, not calibrated SEM resolution, and not a re-evaluation of trained CV at another pitch.

| Pixel pitch (mm) | Signal RMS | Relative RMS difference to 0.0125 mm |
|---:|---:|---:|
| 0.05 | 0.027105 | 0.060 |
| 0.025 | 0.029317 | 0.047 |
| 0.0125 | 0.028520 | 0.000 |

### Error concentration and radial position

These are descriptive post-freeze results. Associations with acquisition, position or clipping are not causal estimates. Surface geometry labels need not produce a detectable contrast under every detector orientation or noise realisation.

| Model | Radial pore offset (mm) | Recall | F1 |
|---|---:|---:|---:|
| cnn_100505003 | -1.05 | 0.889 | 0.130 |
| cnn_100505003 | 0.0 | 0.889 | 0.774 |
| cnn_100505003 | 1.05 | 0.944 | 0.944 |
| median | -1.05 | 0.259 | 0.280 |
| median | 0.0 | 0.222 | 0.293 |
| median | 1.05 | 0.278 | 0.303 |

| Model | Diameter bin (mm) | Recall | F1 |
|---|---|---:|---:|
| cnn_100505003 | 0.500-0.667 | 0.958 | 0.958 |
| cnn_100505003 | 0.667-0.833 | 0.958 | 0.371 |
| cnn_100505003 | 0.833-1.000 | 0.792 | 0.129 |
| median | 0.500-0.667 | 0.250 | 0.286 |
| median | 0.667-0.833 | 0.208 | 0.238 |
| median | 0.833-1.000 | 0.292 | 0.350 |

cnn_100505003: largest false-region count occurs on `test-020`: 412 false regions across its five conditions, including 84 on its clean ring. Its dwell is 0.2957 microseconds. Clean-image high clipping fraction is 0.00000. Hidden-control rings flagged: 3/24.


median: largest false-region count occurs on `test-000`: 2 false regions across its five conditions, including 0 on its clean ring. Its dwell is 16.9751 microseconds. Clean-image high clipping fraction is 0.00000. Hidden-control rings flagged: 0/24.


## Verification and limitations

Pilot gate: `passed`. Geometry anchors, periodicity, displacement-volume identity, hidden/clean image equivalence and visibility refinement were checked. Detailed values are in the saved pilot report.

Nonlocal surface visibility is included under the inherited parallel-direction/constant-local-working-distance detector approximation. No electron trajectories or material-specific information-depth kernel are simulated. Detector response, roughness and acquisition remain uncalibrated. The provided macrograph is qualitative inspiration, not a measured fit.

Sample size supports an exploratory synthetic benchmark, not a power or industrial reliability claim. Adjacent sectors are spatial and do not increase the number of independent weld specimens. Per-scene clipping, signal/noise and visible-area diagnostics are saved in `image_quality.json`; they do not filter scenes or alter thresholds.

Prior failed/development pilot versions remain on disk with source snapshots; only the active campaign linked below supplies these results. No historical model or dataset was overwritten.

## Reproduce

From the repository root:

```powershell
.venv\Scripts\python.exe WeldDefectDetection/run_raised_crown_20261005.py all
.venv\Scripts\python.exe WeldDefectDetection/complete_raised_delivery_20261005.py
```

Active artifacts: `C:\Users\sfleo\OneDrive\CompSciEng\Thesis\Thesis\Codebase2\DataSimulation\data\output\raised_crown_20261005_v5`. See `protocol.json`, `pilot/gates.json`, `cv_protocol.json`, `frozen_models.json`, `dataset_manifest.json`, `results.json`, `robustness_results.json` and `audit.json`.

[SEM/EBW research and modelling decisions](SEM_EBW_DEPTH_AND_PORE_RESEARCH_20261005.md).
[Chronological methods guide](RAISED_CROWN_METHODS_20261005.md).

[Evidence-led next experiment](RAISED_CROWN_NEXT_STEPS_20261005.md).
