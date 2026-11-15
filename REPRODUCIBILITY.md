# Reproduction and scope

## Portable demonstration

`submission.py` reuses the original inference, normalisation, connected-component and box-correction implementations. Its only adapter loads the frozen model assets from paths relative to this folder instead of historical absolute paths. It does not retrain, retune or alter the numerical detector settings.

The model is TinyDefectSegNet, mixed_96, seed 1027001, epoch 20. CNN threshold: 0.9; minimum area: 8; closing: 3. The corrected median uses sigma 1, threshold 16, minimum area 8, closing 3, a two-row shift toward the crown and one-pixel padding. Calibration and normalisation retain the original saved values.

The raw input contract is a finite 160-by-5024 TOPO array with the known ring geometry and radial halo. The scored region is rows 16–143. An arbitrary photograph or height map is not a calibrated TOPO input. External-image preparation in the GUI remains exploratory.

## Evidence map

- Training experiment: `DataSimulation/data/output/count_diversity_20261027_v2/`.
- Main final comparison: `evidence/selected_methods_20261110/summary.json` and the follow-up records.
- Size, position and matching diagnostics: `evidence/posthoc_localisation_20261114/`.
- First median revision and renderer sensitivity: `output/median_exploration_20261115/`.
- Second median revision and controls: `output/median_scale_templates_20261115/`.
- Per-position recall: `output/review/median_recall_by_position_20261115.json`.

The October training-design confirmation and November final A/B confirmation are different evaluations. Do not replace the thesis's Table 5.2 with older numbers in the October report.

## Full campaign reproduction

The campaign scripts and frozen protocols are included for inspection. A complete rerun also requires restoring the original generated populations, dependency archives and historical models referenced by the campaign's hash gates. They are not included in this compact distribution. Historical scripts contain absolute provenance paths and are not claimed to run unchanged after relocation. Do not bypass their frozen-source checks or regenerate into the saved-results directories merely to make a demo run.

`submission.py gui` provides independent synthetic generation outside those research splits. Its outputs are demonstrations, not replacements for the thesis's confirmation data.

## Environments and validation

`requirements-studio.txt` combines simulator and detector dependencies. `requirements-studio-tested.txt` preserves an earlier recorded environment; it is not a newly generated lockfile. The submission validation report records the checks actually run for this delivery, including a separate working-directory test where available.

`PACKAGING_RECORD.json` maps copied files to their original source and hash. `SUBMISSION_MANIFEST.json` identifies the final delivered bytes. The research working tree was not identical to its old Git commit; file hashes, not that commit alone, identify the submitted implementation.
