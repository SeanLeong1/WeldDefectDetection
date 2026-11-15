Median improvement experiment, 15 November 2026

Open REPORT.html for the complete illustrated report. Images are embedded in the HTML.
REPORT.md is the editable text version. results.json and confirmation_results.csv contain
the aggregate results; audit.json records the integrity checks.

The evidence/ folder contains frozen settings, the protocol, full confirmation
predictions and selected diagnostics. scripts/ contains the experiment and analysis
entry points. These scripts require the existing thesis Codebase2 checkout and its
.venv Python environment, historical datasets, frozen CNN checkpoint and dependencies.
The bundle is an evidence handoff, not a self-contained training environment.

Large raw image arrays and complete validation component caches remain in:
DataSimulation/data/output/median_exploration_20261115_v1

Use the frozen revised median on a compatible raw registered TOPO array:
.venv\Scripts\python.exe WeldDefectDetection\apply_revised_median_20261115.py INPUT.npy --output PREDICTIONS.json

All experiments are synthetic. No measured-image validity or industrial deployment
claim is established. Renderer changes are sensitivity assumptions, not calibrated
physical corrections. Do not tune on the fresh confirmation outcomes.
