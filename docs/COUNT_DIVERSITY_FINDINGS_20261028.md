# Count-noise study: findings and claim limits

Campaign: `count_diversity_20261027_v2`. Read this interpretation with the [generated results report](COUNT_DIVERSITY_RESULTS_20261027.md). The [execution record](COUNT_DIVERSITY_EXECUTION_20261027.md) defines the protocol and final verification requirements.

## Main comparison

The validation-selected 96-background mixed-noise CNN clearly exceeds its corresponding median baseline on fresh synthetic confirmation. Detection F1 combines precision and recall: `F1 = 2 TP / (2 TP + FP + FN)`. The scorer requires one-to-one matching of circular bounding boxes at intersection-over-union of at least 0.5. This measures defect detection, rather than pixel accuracy. Pixel IoU separately measures overlap between predicted and true defect masks.

| Confirmation counts | Mixed CNN F1 | Mixed median F1 | Paired F1 difference, 95% background interval |
|---|---:|---:|---|
| 1x | 0.7737 | 0.1267 | +0.647 [0.535, 0.744] |
| 4x | 0.8853 | 0.1781 | +0.707 [0.595, 0.803] |
| 64x | 0.9091 | 0.1917 | +0.717 [0.612, 0.811] |

The mixed CNN flags 5/96 clean acquisitions at 1x and 0/96 at both 4x and 64x. Each count level contains two acquisitions of each of 48 clean backgrounds; 96 does not mean 96 independent geometries. The intervals resample backgrounds and retain paired variants. They condition on the fitted models and do not include training-seed uncertainty. The report lists seed ranges separately. These exploratory intervals do not adjust for multiple comparisons.

Evidence: [saved scene metrics](../DataSimulation/data/output/count_diversity_20261027_v2/results.json), [paired comparisons](../DataSimulation/data/output/count_diversity_20261027_v2/paired_comparisons.json), and [metric implementation](../WeldDefectDetection/circular_study/metrics.py).

## What the experiment establishes

- Lower acquisition noise helps the mixed CNN: its selected-model F1 increases from 0.774 to 0.909. The median does not approach the CNN in this experiment, even at 64x counts.
- Training only at 64x counts transfers poorly to 1x: the selected specialist CNN falls from 0.920 to 0.347. Training-domain coverage matters.
- Increasing mixed-training backgrounds from 24 to 96 improves 1x F1 by 0.093, with a paired interval of [0.007, 0.172]. The corresponding intervals at 4x and 64x include zero.
- Mixed training does not show a clear advantage over 1x-only training with 96 backgrounds: all three paired difference intervals include zero. The 4x specialist has numerically higher F1 than the mixed model at all three levels. This observation does not authorize selecting or retuning a model on confirmation.
- The selected mixed CNN has pooled recall of 0.667 for 0.15–0.30 mm pores, versus 0.912 for 0.60–1.0 mm pores. Small defects remain a limitation. These size summaries pool all count levels.

Evidence: the same saved metrics and paired comparisons above. The [frozen selections](../DataSimulation/data/output/count_diversity_20261027_v2/frozen_models.json) identify the models independently of confirmation performance.

## Limits that must accompany the thesis claim

Every selected median uses threshold 12, the largest threshold in the declared grid, and minimum area 15, the largest tested area cutoff. The experiment therefore compares the CNN with a validation-selected baseline within a finite search, not a globally optimal median. Some ground-truth pore footprints contain fewer than 15 scored pixels. The area cutoff applies to predicted components, so this does not prove that every such pore must fail, but size and postprocessing remain relevant constraints. A broader baseline study would require a separately declared development protocol and fresh confirmation; this study leaves its frozen selections unchanged.

The simulator changes effective count conversion at fixed dwell. It does not calibrate physical primary-electron counts or establish beam-current/dwell equivalence. The results concern simulated surface-opening pores under known ring coordinates. They do not establish buried-defect detection, real-image transfer, automatic localisation, or suitability for arbitrary input images.

Evidence: [protocol and search grid](../WeldDefectDetection/count_study/campaign.py), [selected settings](COUNT_DIVERSITY_RESULTS_20261027.md), and [execution record](COUNT_DIVERSITY_EXECUTION_20261027.md).

## Next steps

Retain the frozen synthetic comparison as the thesis result. Integrate its tables, figures and limitations into the thesis. For a usable inference package, define compatible raw-image inputs and export masks, regions and metadata. Then address image-estimated ring localisation and measured-image calibration/transfer. The two-layer CNN remains unimplemented and outside this study.
