# Effective-count and training-diversity study, 27 October 2026

Campaign: `count_diversity_20261027_v2`. All values below come from newly generated synthetic confirmation. No physical validation.

## Completed design

96 training, 32 validation and 48 confirmation backgrounds; each has clean, intrinsic-pore and keyhole-pore conditions at 1x/4x/64x effective counts, with two acquisition repeats. Total: 3,168 scenes across 176 independent backgrounds. All paired variants remain in the same split.
Count multiplication changes effective count conversion at fixed dwell, geometry and other acquisition parameters. It does not specify calibrated primary electrons. The original broad dwell/profile distribution remains, so these are relative count levels rather than three absolute noise amplitudes.
All six arms use the same TinyDefectSegNet architecture, 20 epochs and 1,200 patch draws per epoch. Three seeds per arm, epochs 10/20 and detection operating points enter validation-only selection. Specialists select on their own count level; mixed arms maximize mean validation F1 across levels. The mixed/single and 24/96 comparisons preserve total training updates.
The shared normalization uses the nested 24 training backgrounds at all count levels. Consequently, single-noise arms restrict supervised weight training, not exposure of the global normalization to the other count levels.
Each median searches six Gaussian scales, six thresholds, three area cutoffs and two closing settings, with clean-training calibration. All final choices freeze before confirmation generation.

## Fresh confirmation results

| Training arm / method | F1 at 1x | F1 at 4x | F1 at 64x |
|---|---:|---:|---:|
| mixed_24_cnn | 0.6808 | 0.8527 | 0.9043 |
| mixed_24_median | 0.1090 | 0.1646 | 0.1822 |
| mixed_96_cnn | 0.7737 | 0.8853 | 0.9091 |
| mixed_96_median | 0.1267 | 0.1781 | 0.1917 |
| retained_cnn | 0.5511 | 0.6448 | 0.6613 |
| retained_median | 0.1914 | 0.1896 | 0.1702 |
| single1_24_cnn | 0.6595 | 0.7204 | 0.7769 |
| single1_24_median | 0.2321 | 0.2321 | 0.2357 |
| single1_96_cnn | 0.7990 | 0.8667 | 0.9024 |
| single1_96_median | 0.2085 | 0.1969 | 0.1771 |
| single4_96_cnn | 0.8276 | 0.8953 | 0.9182 |
| single4_96_median | 0.0432 | 0.2074 | 0.2341 |
| single64_96_cnn | 0.3475 | 0.8131 | 0.9198 |
| single64_96_median | 0.0014 | 0.0483 | 0.1635 |

## Training-seed variability

| Arm | Count | Minimum seed F1 | Maximum seed F1 |
|---|---:|---:|---:|
| single1_96 | 1 | 0.7421 | 0.7990 |
| single1_96 | 4 | 0.8150 | 0.8667 |
| single1_96 | 64 | 0.7921 | 0.9024 |
| single4_96 | 1 | 0.6619 | 0.8276 |
| single4_96 | 4 | 0.8557 | 0.8953 |
| single4_96 | 64 | 0.8984 | 0.9182 |
| single64_96 | 1 | 0.0962 | 0.3475 |
| single64_96 | 4 | 0.6872 | 0.8131 |
| single64_96 | 64 | 0.9025 | 0.9365 |
| mixed_96 | 1 | 0.6819 | 0.7754 |
| mixed_96 | 4 | 0.8626 | 0.8853 |
| mixed_96 | 64 | 0.9011 | 0.9091 |
| single1_24 | 1 | 0.2429 | 0.6595 |
| single1_24 | 4 | 0.2587 | 0.7204 |
| single1_24 | 64 | 0.2465 | 0.7769 |
| mixed_24 | 1 | 0.4589 | 0.6808 |
| mixed_24 | 4 | 0.4843 | 0.8527 |
| mixed_24 | 64 | 0.5065 | 0.9043 |

![Train/test matrix](figures/count_diversity_20261027/train_test_matrix.png)

## Selected operating points

| Arm | CNN seed / epoch | CNN threshold / area / closing | Median sigma / threshold / area / closing |
|---|---|---|---|
| single1_96 | 1027001 / 20 | 0.5 / 8 / 1 | 2.0 / 12.0 / 15 / 3 |
| single4_96 | 1027001 / 20 | 0.65 / 8 / 3 | 0.5 / 12.0 / 15 / 3 |
| single64_96 | 1027002 / 20 | 0.65 / 8 / 3 | 0.0 / 12.0 / 15 / 3 |
| mixed_96 | 1027001 / 20 | 0.9 / 8 / 3 | 1.0 / 12.0 / 15 / 3 |
| single1_24 | 1027003 / 20 | 0.2 / 15 / 3 | 1.0 / 12.0 / 15 / 3 |
| mixed_24 | 1027001 / 20 | 0.5 / 8 / 3 | 1.0 / 12.0 / 15 / 3 |

## Precision, recall and clean alarms

| Method | Counts | Precision | Recall | Clean rings flagged | False regions on clean | Pixel IoU |
|---|---:|---:|---:|---:|---:|---:|
| mixed_24_cnn | 1 | 0.620 | 0.755 | 11/96 | 17 | 0.534 |
| mixed_24_cnn | 4 | 0.846 | 0.859 | 4/96 | 5 | 0.618 |
| mixed_24_cnn | 64 | 0.924 | 0.885 | 0/96 | 0 | 0.646 |
| mixed_24_median | 1 | 0.071 | 0.234 | 26/96 | 154 | 0.127 |
| mixed_24_median | 4 | 0.125 | 0.240 | 22/96 | 68 | 0.135 |
| mixed_24_median | 64 | 0.147 | 0.240 | 24/96 | 50 | 0.109 |
| mixed_96_cnn | 1 | 0.782 | 0.766 | 5/96 | 5 | 0.637 |
| mixed_96_cnn | 4 | 0.907 | 0.865 | 0/96 | 0 | 0.734 |
| mixed_96_cnn | 64 | 0.934 | 0.885 | 0/96 | 0 | 0.766 |
| mixed_96_median | 1 | 0.088 | 0.224 | 22/96 | 112 | 0.139 |
| mixed_96_median | 4 | 0.146 | 0.229 | 17/96 | 50 | 0.147 |
| mixed_96_median | 64 | 0.165 | 0.229 | 18/96 | 37 | 0.117 |
| retained_cnn | 1 | 0.507 | 0.604 | 13/96 | 18 | 0.521 |
| retained_cnn | 4 | 0.624 | 0.667 | 3/96 | 9 | 0.645 |
| retained_cnn | 64 | 0.678 | 0.646 | 1/96 | 1 | 0.672 |
| retained_median | 1 | 0.235 | 0.161 | 1/96 | 1 | 0.163 |
| retained_median | 4 | 0.230 | 0.161 | 0/96 | 0 | 0.175 |
| retained_median | 64 | 0.204 | 0.146 | 0/96 | 0 | 0.181 |
| single1_24_cnn | 1 | 0.680 | 0.641 | 6/96 | 7 | 0.478 |
| single1_24_cnn | 4 | 0.744 | 0.698 | 2/96 | 3 | 0.545 |
| single1_24_cnn | 64 | 0.825 | 0.734 | 1/96 | 1 | 0.574 |
| single1_24_median | 1 | 0.337 | 0.177 | 0/96 | 0 | 0.127 |
| single1_24_median | 4 | 0.337 | 0.177 | 0/96 | 0 | 0.128 |
| single1_24_median | 64 | 0.333 | 0.182 | 1/96 | 1 | 0.129 |
| single1_96_cnn | 1 | 0.791 | 0.807 | 4/96 | 6 | 0.664 |
| single1_96_cnn | 4 | 0.854 | 0.880 | 2/96 | 4 | 0.750 |
| single1_96_cnn | 64 | 0.914 | 0.891 | 0/96 | 0 | 0.789 |
| single1_96_median | 1 | 0.278 | 0.167 | 0/96 | 0 | 0.167 |
| single1_96_median | 4 | 0.241 | 0.167 | 3/96 | 6 | 0.161 |
| single1_96_median | 64 | 0.196 | 0.161 | 9/96 | 14 | 0.123 |
| single4_96_cnn | 1 | 0.843 | 0.812 | 0/96 | 0 | 0.651 |
| single4_96_cnn | 4 | 0.900 | 0.891 | 1/96 | 1 | 0.776 |
| single4_96_cnn | 64 | 0.930 | 0.906 | 0/96 | 0 | 0.814 |
| single4_96_median | 1 | 0.024 | 0.224 | 10/96 | 545 | 0.057 |
| single4_96_median | 4 | 0.290 | 0.161 | 0/96 | 0 | 0.132 |
| single4_96_median | 64 | 0.327 | 0.182 | 0/96 | 0 | 0.134 |
| single64_96_cnn | 1 | 0.238 | 0.641 | 19/96 | 118 | 0.417 |
| single64_96_cnn | 4 | 0.789 | 0.839 | 4/96 | 8 | 0.708 |
| single64_96_cnn | 64 | 0.945 | 0.896 | 0/96 | 0 | 0.792 |
| single64_96_median | 1 | 0.001 | 0.229 | 55/96 | 20580 | 0.001 |
| single64_96_median | 4 | 0.026 | 0.312 | 39/96 | 710 | 0.065 |
| single64_96_median | 64 | 0.117 | 0.271 | 28/96 | 94 | 0.097 |

## Paired F1 differences

| Counts | Comparison | Difference | 95% background interval |
|---:|---|---:|---|
| 1 | single1_96_cnn minus single1_96_median | +0.591 | [+0.449, +0.732] |
| 1 | mixed_96_cnn minus mixed_96_median | +0.647 | [+0.535, +0.744] |
| 1 | mixed_96_cnn minus single1_96_cnn | -0.025 | [-0.106, +0.053] |
| 1 | mixed_96_cnn minus mixed_24_cnn | +0.093 | [+0.007, +0.172] |
| 1 | single1_96_cnn minus single1_24_cnn | +0.139 | [+0.044, +0.242] |
| 1 | mixed_96_median minus single1_96_median | -0.082 | [-0.147, -0.013] |
| 1 | mixed_96_median minus mixed_24_median | +0.018 | [-0.000, +0.043] |
| 4 | single4_96_cnn minus single4_96_median | +0.688 | [+0.559, +0.805] |
| 4 | mixed_96_cnn minus mixed_96_median | +0.707 | [+0.595, +0.803] |
| 4 | mixed_96_cnn minus single1_96_cnn | +0.019 | [-0.059, +0.096] |
| 4 | mixed_96_cnn minus mixed_24_cnn | +0.033 | [-0.042, +0.113] |
| 4 | single1_96_cnn minus single1_24_cnn | +0.146 | [+0.055, +0.247] |
| 4 | mixed_96_median minus single1_96_median | -0.019 | [-0.072, +0.038] |
| 4 | mixed_96_median minus mixed_24_median | +0.014 | [-0.012, +0.046] |
| 64 | single64_96_cnn minus single64_96_median | +0.756 | [+0.665, +0.837] |
| 64 | mixed_96_cnn minus mixed_96_median | +0.717 | [+0.612, +0.811] |
| 64 | mixed_96_cnn minus single1_96_cnn | +0.007 | [-0.068, +0.084] |
| 64 | mixed_96_cnn minus mixed_24_cnn | +0.005 | [-0.065, +0.083] |
| 64 | single1_96_cnn minus single1_24_cnn | +0.126 | [+0.048, +0.225] |
| 64 | mixed_96_median minus single1_96_median | +0.015 | [-0.042, +0.075] |
| 64 | mixed_96_median minus mixed_24_median | +0.010 | [-0.019, +0.040] |

Intervals resample independent backgrounds, retaining all count, defect and noise-repeat variants together. They condition on fitted models; the saved results separately retain all three training seeds. These exploratory multiple comparisons are not multiplicity-adjusted confirmatory claims.

## Noise and examples

![Measured noise](figures/count_diversity_20261027/measured_noise.png)
Repeat-difference RMS includes acquisition variability. It is not calibrated physical electron noise. Nonlinear TOPO magnitude and unchanged electronic noise can prevent the full image from following the Poisson-only 1/sqrt(count) law.

![Matched example](figures/count_diversity_20261027/matched_clean_pore.png)

The fixed .2 mm pore diagnostic includes noiseless, shot-only and full acquisition images. It never enters training or model selection. Its saved predictions diagnose a local ceiling; one geometry cannot establish population performance.

## Limits and next steps

The comparison addresses relative count statistics and training diversity under declared ring coordinates. It does not resolve ring localisation, real-image transfer, detector calibration, buried-pore visibility, or measured beam-current/dwell equivalence.
We retained all historical campaigns and did not implement the proposed two-layer CNN. A size study remains optional. The next scientific priorities are real-image calibration/transfer and localisation robustness. The next delivery task is independent compatible-image import and export.
Small-pore sampling extends the earlier .5-1.0 mm training distribution, so old-model performance here is transfer to a joint size/background/count challenge. It is not an isolated causal comparison of training noise.

## Evidence and reproduction

Artifacts: `DataSimulation/data/output/count_diversity_20261027_v2/`: protocol, noise checks, method checks, train/validation/test indexes, training histories and checkpoints, frozen manifest, all-seed predictions, paired comparisons, diagnostic results and audit.
Run `.venv/Scripts/python.exe WeldDefectDetection/run_count_study_20261027.py all`. Completed stages verify and reuse their saved outputs. Historical code and data remain read-only.
Prediction timers include image reading, normalization, network/median application and score-file writing. They exclude metric selection and bootstrap calculations. They are workflow timings, not pure model latency.
