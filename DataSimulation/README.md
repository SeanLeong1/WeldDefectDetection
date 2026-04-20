# BSE Renderer DataSimulation

Synthetic EBW weld defect data generation framework with modular physics, geometry, rendering, noise, validation, and output pipelines.

## Structure

- `config/` YAML domain configs (`beam`, `material`, `detector`, `defect`, `noise`)
- `bse_renderer/` core package
- `entrypoints/` CLI entrypoints
- `tests/` unit/integration tests
- `main.py` top-level launcher

## CLI

```bash
python main.py gui --config-dir config
python main.py single --config-dir config --seed 42 --image-size 256 --preview
python main.py single --config-dir config --avg-z 27 --preview
python main.py single --config-dir config --defect-plan "keyhole_pore(diameter_mm=1.2,depth_mm=0.8,count=2);crack_solidification(length_mm=3.0,width_mm=0.1,depth_mm=0.4)" --preview
python main.py batch --config-dir config --num-images 200 --output-dir data/output --workers 1
python main.py validation
python main.py validation --casino-exe "C:/path/to/casino_adapter.exe" --casino-calibrate --casino-budget medium
python main.py validation --standards-t-mm 12 --standards-s-mm 6 --contrast-threshold 0.025
```

## Manual Defect Manufacturing

- GUI: use the `Manufactured Defects (EBW-relevant)` checklist in the `Run` tab.
- For each defect: tick it, set `Count`, `Size`, `Depth` (when applicable), then preview.
- Use `Select All`, `Clear All`, and `Obvious Sizes` quick buttons.
- For clear morphology inspection, leave `Apply noise in preview` unchecked (default).
- CLI: use `--defect-plan` in `single` mode if you want scripted/manual plans.
- Syntax: `defect_type(param=value, ...)`, separated by `;`.
- Optional `count` repeats one defect definition: `keyhole_pore(diameter_mm=1.0,depth_mm=0.6,count=3)`.

## Average Atomic Number Material Override

- GUI: enable `Use average atomic number override` and set `Zavg`.
- CLI: pass `--avg-z`.
- The renderer estimates `a`, `rho`, `b`, and `n` via interpolation over embedded reference materials plus your configured presets.

## Output Artifacts

- `images/compo/<split>/*.png` (16-bit)
- `images/topo/<split>/*.png` (16-bit)
- `images_8bit/...` preview images
- `masks/<split>/*.png` segmentation masks
- `labels/yolo/<split>/*.txt` YOLO labels
- `labels/coco/annotations.json` COCO labels
- `manifest.json` full metadata traceability
- `dataset.yaml` training dataset descriptor

## Validation Output Bundle

`validation` now writes a report bundle under:

- `data/output/validation_reports/casino_validation_<timestamp>/validation_report.json`
- `.../threshold_slices.csv`
- `.../threshold_slices.png` (if `matplotlib` is available)
- `.../regimes/regime_sweep_report.json`
- `.../calibration/renderer_calibration.json` (when `--casino-calibrate` is enabled)

Key validation controls:

- `--casino-exe` path (or `BSE_CASINO_EXE`)
- `--casino-calibrate` and `--casino-calibration-cases`
- `--casino-budget {fast,medium,high}`
- `--standards-t-mm`, `--standards-s-mm`, `--contrast-threshold`
- Optional scaffold adapter for wiring checks: `tools/casino/casino_adapter.cmd`
- External adapter mode supports `BSE_CASINO_EXTERNAL_RUNNER_SCRIPT`/`BSE_CASINO_EXTERNAL_RUNNER_ARGS` (see `tools/casino/README.md`)
- AutoHotkey-driven CASINO automation runner is provided in `tools/casino/casino_runner_template.py` with profile template `tools/casino/casino_ahk_profile.example.json`

## Optional Dependencies

- `pyvista` and its rendering backend are required for interactive 3D mesh preview.
- `trimesh` is required for CAD mesh import (`STL/OBJ/PLY`).
- CASINO validation adapter is optional. `python main.py validation` runs standards/regime reporting with the fast renderer and reports CASINO probe status if `--casino-exe` (or `BSE_CASINO_EXE`) is not available.
