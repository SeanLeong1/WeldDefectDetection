"""Promote only a completed, audited study with verified GUI workflows."""
from . import campaign as c

def run():
    audit=c.read(c.OUT/'audit.json');assert audit['status']=='passed'
    assert (c.OUT/'gui_checks.json').exists()
    result=c.read(c.OUT/'results.json');selected=result['results'][result['selected']]['metrics'];median=result['results']['median']['metrics']
    for module in ('DataSimulation','WeldDefectDetection'):
        path=c.ROOT/module/'README.md'
        archive=c.ROOT/'docs/archive'/f'{module}_README_before_raised_20261005.md'
        if not archive.exists():archive.write_bytes(path.read_bytes())
        path.write_text(f'''# {module}: current raised-crown workflow

The current circular-only study is `raised_crown_20261005_v5`. Use the [current project guide](../docs/PROJECT_CURRENT.md), [results](../docs/RAISED_CROWN_RESULTS_20261005.md), [methods](../docs/RAISED_CROWN_METHODS_20261005.md) and [SEM/EBW research](../docs/SEM_EBW_DEPTH_AND_PORE_RESEARCH_20261005.md).

The 3 mm crown uses a separate versioned renderer and models. Earlier emission assumptions, input dimensions and model defaults described below apply to the historical workflows. They are preserved for reproduction and are not the new raised-crown contract.

## Historical module documentation, preserved

'''+archive.read_text(encoding='utf-8'),encoding='utf-8')
    pipeline=c.ROOT/'DataSimulation/PROJECT_PIPELINE_CURRENT.md'
    archive=c.ROOT/'docs/archive/PROJECT_PIPELINE_CURRENT_before_raised_20261005.md'
    if not archive.exists():archive.write_bytes(pipeline.read_bytes())
    pipeline.write_text('''# Current pipeline and preserved CASINO project history

The authoritative current workflow is the [raised-crown project guide](../docs/PROJECT_CURRENT.md). The raised-crown revision has its own geometry, uncalibrated Lambertian response, nonlocal visibility, models and results. The earlier constant-B and crop-based workflows below remain historical evidence; their references to current models are relative to their recorded dates.

Read the [raised-crown methods](../docs/RAISED_CROWN_METHODS_20261005.md), [results](../docs/RAISED_CROWN_RESULTS_20261005.md), and [SEM depth research](../docs/SEM_EBW_DEPTH_AND_PORE_RESEARCH_20261005.md).

## Preserved earlier project guide

'''+archive.read_text(encoding='utf-8'),encoding='utf-8')
    text=f'''# Current project: raised circular crown, 5 October 2026

The current research workflow is `{c.OUT.name}`: a circular weld with a 3 mm single-peaked crown, a solid 7 mm-deep fusion region, a 2.8 mm fusion width at the original surface and 1.2 mm width at 5 mm depth. Crown footprint is provisionally 2.8 mm. Plate thickness is outside this surface-study scope.

This is a synthetic study with explicit uncalibrated Lambertian response and nonlocal geometric visibility. It is not a physically calibrated SEM/EBW process simulation. Published electron-depth numbers were researched but not implemented. The existing range diagnostic is not a BSE visibility cutoff.

## Completed evidence

Independent backgrounds: {audit['backgrounds']}. Scenes: {audit['counts']}. Confirmation was generated after model/threshold freeze. Three CNN seeds were trained with the same base24 architecture; the neighbouring median was recalibrated from clean training data.

Selected CNN confirmation F1: **{selected['f1']:.3f}**. Median confirmation F1: **{median['f1']:.3f}**. These results apply to the declared raised-crown synthetic population and must not be compared causally with older campaigns having different image formation and defect laws.

- [Results, robustness and limits](RAISED_CROWN_RESULTS_20261005.md)
- [Chronological geometry, CNN layers and median guide](RAISED_CROWN_METHODS_20261005.md)
- [SEM/BSE depth and EBW pore research](SEM_EBW_DEPTH_AND_PORE_RESEARCH_20261005.md)
- [Evidence-led next experiment](RAISED_CROWN_NEXT_STEPS_20261005.md)
- [Revision plan and decisions](RAISED_CROWN_FUSION_REVISION_PLAN_20261005.md)
- [Previous current-project guide, preserved](archive/PROJECT_CURRENT_before_raised_20261005.md)

## Run the GUI

Use `Launch Weld Vision.cmd` or `launch_weld_studio.py`. After the study audit and GUI checks exist, the launcher opens the raised-crown workflow. `launch_raised_crown.py` opens it explicitly. The preserved recessed circular and older straight-weld workflows remain available on labelled historical tabs.

The new tab distinguishes surface-opening pores, assumed volume-displacement bulges and enclosed/no-surface-response controls. It shows nominal fusion geometry, actual surface height and rendered TOPO with predictions. It rejects incompatible geometry metadata and image dimensions. Inference reads raw TOPO and never uses truth labels. New examples are saved separately from research splits.

## Reproduce

From the repository root, run:

```powershell
.venv\\Scripts\\python.exe WeldDefectDetection/run_raised_crown_20261005.py all
.venv\\Scripts\\python.exe WeldDefectDetection/complete_raised_delivery_20261005.py
```

`run_raised_delivery_20261005.py` schedules generation across disjoint background workers. It preserves the same scene seeds and renderer. Full research replay needs the workspace's research artifacts and tested dependencies. Portable inference bundles retain the selected runtime, example data, models and historical choices; they do not contain every training image.

The newest raised-crown bundle is identified by `data/releases/CURRENT_RAISED_RELEASE.json` after its relocation checks pass. The earlier `CURRENT_RELEASE.json` is retained unchanged as the historical 4 October release record, not the newest raised-crown bundle.

## Version history

| Evidence stream | Status |
|---|---|
| Straight-weld campaigns, September | Historical, unchanged |
| Circular recessed study v4 and robustness, 4 October | Historical completed benchmark, unchanged |
| Raised-crown development v1 | Pilot exposed a noisy-realisation comparison error; no trained model |
| Raised-crown development v2 | Numerical local-response pilot; nonlocal shadowing omission identified |
| Raised-crown development v3 | Ray visibility and refinement pilot |
| Raised-crown development v4 | Cached visibility verification and pre-training data; volume-coordinate correction required |
| Raised-crown study v5 | Current corrected campaign; unaffected development images reused with explicit receipts, displacement scenes regenerated |

The older source snapshots, data and models remain available. New results do not turn prior recessed-weld scores into raised-crown evidence.

## Scientific boundaries

The fusion region is solid metal, not an open keyhole. Prescribed pore-volume displacement is a counterfactual model, not a universal prediction that a buried pore must cause a bump. Enclosed pores without a surface response are negative surface controls. Their simulated invisibility is not a measured SEM information-depth result. Further physical claims require measured crown profiles, appropriate electron/detector calibration and independent real TOPO images.
'''
    (c.ROOT/'docs/PROJECT_CURRENT.md').write_text(text,encoding='utf-8')
    (c.ROOT/'README.md').write_text('''# Weld Vision: raised circular crown

The current synthetic research study uses an 80 mm circular weld with a single-peaked 3 mm crown and a solid fusion outline reaching 7 mm below the original surface. It compares a compact CNN with neighbouring-median detection. Published SEM penetration numbers were researched but not implemented.

## Start

Double-click **Launch Weld Vision.cmd**, or run:

```powershell
.venv\\Scripts\\python.exe launch_weld_studio.py
```

The first tab is the raised-crown workflow after its audit and GUI checks pass. It generates surface-opening pores, assumed volume-displacement bulges and enclosed/no-surface-response controls, and compares CNN and median predictions. Preserved recessed circular and straight-weld workflows remain on labelled historical tabs.

Read [current status and version history](docs/PROJECT_CURRENT.md), [results and robustness](docs/RAISED_CROWN_RESULTS_20261005.md), [chronological methods and CNN layers](docs/RAISED_CROWN_METHODS_20261005.md), and [SEM/EBW research](docs/SEM_EBW_DEPTH_AND_PORE_RESEARCH_20261005.md).

## Input and scientific contract

- The raised-crown workflow accepts its generated raw NPZ files with matching geometry metadata and checksum: 160 x 5024 annuli, radial pitch 0.05 mm, nominal radius 40 mm. The central 128 rows are scored.
- TOPO is detector contrast magnitude, not surface height. One crown peak can create two bright slope bands. The GUI shows height and TOPO separately.
- Inference uses raw TOPO and does not read truth labels. Ring-localisation robustness is a research evaluation, not a general unknown-image import feature.
- Surface signals use an explicit uncalibrated Lambertian response and nonlocal geometric visibility. The previous B fit cannot support this crown's steep slopes.
- Prescribed bulges are a volume-displacement hypothesis, not a universal physical consequence of an enclosed pore. No electron trajectories or calibrated subsurface sensitivity are simulated.

## Reproduction and verification

```powershell
.venv\\Scripts\\python.exe WeldDefectDetection/run_raised_crown_20261005.py all
.venv\\Scripts\\python.exe WeldDefectDetection/complete_raised_delivery_20261005.py
```

The parallel delivery runner is `WeldDefectDetection/run_raised_delivery_20261005.py`. Related backgrounds remain grouped across conditions; model selection is validation-only and confirmation generation follows the model freeze. Full replay requires the research artifacts and tested dependency environment.

The portable inference release is identified by `data/releases/CURRENT_RAISED_RELEASE.json`. It contains models, examples and reports rather than all research images. Relocation checks use the existing environment; a fresh dependency installation is not claimed.

## Historical evidence

Historical datasets, checkpoints, source snapshots, reports and releases remain in place. The earlier release pointer `CURRENT_RELEASE.json` still identifies the preserved 4 October release. [The previous README](docs/archive/README_before_raised_20261005.md) and [previous current-project guide](docs/archive/PROJECT_CURRENT_before_raised_20261005.md) record that workflow.

New scores apply to the declared synthetic raised-crown population. They do not establish measured SEM realism, real-image transfer, hidden-pore detection or manufacturing acceptance.
''',encoding='utf-8')
    print('Current-project guide promoted; historical guide retained',flush=True)

if __name__=='__main__':run()
