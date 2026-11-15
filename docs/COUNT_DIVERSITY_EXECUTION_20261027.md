# Count-study execution record

Submission extract: completed-study facts retained; obsolete agent/session monitoring instructions omitted. The frozen JSON protocol and audit records remain unchanged.

## Verified completion, 28 October 2026 Sydney time

The controller completed at 2026-10-27 19:16:42 UTC. A fresh-process report refresh subsequently completed, followed by confirmation of the independent final audit, 95 report-table rows checked against saved results, and visual inspection of all three PNG figures. The independent audit checked 3,696 data-file hashes, split-disjoint background/acquisition seeds, all six development selections, fresh prediction parity, original-metric validation replay, and all 26 confirmation summaries over 864 scenes each. The full dataset contains 3,168 scenes across the required 96/32/48 backgrounds. Frozen selections predate the earliest confirmation raw file (13:33:43 versus 13:35:40 UTC). No model or operating point changed during final verification.

Receipts: `independent_final_audit.json` and `final_delivery_verification.json` in the v2 campaign directory. Read [results](COUNT_DIVERSITY_RESULTS_20261027.md) and [findings and limits](COUNT_DIVERSITY_FINDINGS_20261028.md).

## Implemented design

| Item | Implementation |
|---|---|
| Counts | 1x, 4x and 64x effective count conversion at unchanged dwell/profile/geometry |
| Split | 96 training, 32 validation, 48 confirmation backgrounds |
| Variants | Clean, intrinsic surface pore, keyhole surface pore; three count settings; two acquisition repeats |
| Size | 3,168 full-ring TOPO scenes across 176 independent backgrounds |
| Geometry diversity | Pore diameter bins .15-.30, .30-.60 and .60-1.0 mm, depth .15-.35 mm, radial/angle and normal-crown variation |
| CNN | Existing 262,777-parameter TinyDefectSegNet, single TOPO channel; 20 epochs x 1,200 draws, three seeds per arm |
| Six arms | 96-background specialists at 1x/4x/64x; 96-background mixed; 24-background 1x and mixed controls |
| CNN selection | Epoch 10 or 20, seed, threshold, minimum area and closing, using validation only |
| Median selection | Six Gaussian scales, six thresholds, three area cutoffs, two closing settings; clean-training residual calibration |
| Mixed selection | Mean validation F1 over count levels, then pooled pixel IoU and fewer false regions |
| Shared normalisation | Nested 24 training backgrounds across all counts; specialists restrict supervised training, not normalisation exposure |
| Confirmation | Generate only after all six arms and all seeds freeze; evaluate every seed and retained original CNN/median |
| Statistics | Full-ring periodic box IoU >= .5, precision/recall/F1/pixel IoU, clean alarms, size breakdown, paired background intervals |
| Diagnostics | Fixed .2 mm pore, noiseless/shot-only/full acquisition; excluded from selection |

The main count multiplier is an effective count parameter, not calibrated primary electrons. Other acquisition noise remains. Baseline dwell and profile variation means these are relative count levels, not three uniform absolute noise amplitudes. The old-model control runs on the fresh confirmation after development freeze, avoiding feedback into model selection.
