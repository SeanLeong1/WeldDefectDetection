# Correcting B using the existing CASINO data

## What changed

The earlier interpretation of the reliability failures was incomplete. Two selection errors made the historical high-tilt fits misleading. They can be corrected using the retained summary exports, without another simulation campaign.

1. The historical parser fitted one entrance plane across all scan points in a file. Each retained `70deg-r*` file contains a landing patch on a roughly 20-degree face as well as a patch on the intended 70-degree face. Combining them produced the misleading 67.889-degree estimate.
2. The historical fit selected only histories labelled `Backscattered`. In these exports the global exit classification splits electrons according to the global direction: some `Transmitted` histories actually leave outward through the same tilted surface that the beam entered. Omitting them increasingly truncates the local exit distribution as tilt increases.

The correction identifies the entrance plane for each scan-point block and selects outgoing electrons by geometry and energy, using both recorded exit classes. The 20-degree patches in the nominal 70-degree files remain recorded but are excluded from the nominal 70-degree calibration. They are not silently counted as new independent 20-degree runs.

All original trajectories, binary CASINO files, historical reports, synthetic datasets and trained checkpoints were preserved. The renderer remains on its existing explicitly selected emission mode. The new fits are not silently loaded into it.

## Existing data found and used

The two supplied directory trees contain CASINO binary projects and exports. The canonical repeated tilt campaign already exists inside `Codebase2/DataSimulation/data/output/casino_reruns/parametric_model_tilt_304ss_60kev_combined_0to70_fit/cas`.

The analysis used all 48 summary exports in that canonical campaign, with 74 scan-point blocks and 1,479,926 exported primary histories. Other copies and older Obsidian exports were not added as independent repeats. The summaries already contain boundary information, so there was no need to decode or copy the large `.cas` histories.

| Artifact | Size and purpose |
|---|---|
| Existing 48 text summaries | 597.2 MB total; read without loading complete files into memory |
| New compressed boundary caches | 116.7 MB total; first/final rows, IDs, exit classes and metadata retained |
| Selected angular samples | 7.65 MB; 466,055 usable same-surface exits |
| Angular reference table | 19.6 kB; conditional angular probability masses for later model comparison |

These are additional compact artifacts, not a claim that original disk space was reclaimed. No original file was deleted. Future angular fitting needs boundary information rather than every internal collision, provided the original simulation settings and provenance are retained.

The legacy text parser was also changed to iterate through the file and keep at most the first and final collision row of each history. Its memory now scales with history count rather than the number of stored collisions. The new analysis uses compressed caches and verifies source SHA-256 before reusing them.

## How surface selection works

For each scan-point block:

1. Fit an entrance plane to first positions and orient its normal toward the source.
2. Refine that plane using nearby boundary endpoints. The small entrance spot spans nanometres, so rounded coordinates can produce a slightly inaccurate slope when extrapolated over micrometres. Refinement may change the normal by no more than 0.1 degrees and must keep the entrance points on the plane.
3. Require the measured incidence to match the declared tilt within 0.1 degrees. Preserve rejected patches and their counts in the selection report.
4. For histories with an exported final boundary row, require that endpoint to lie within 1 nm of the entrance plane, its normalized direction to satisfy `u dot n > 0`, and its energy to exceed 50 eV. Consider both `Backscattered` and `Transmitted` labels.
5. Express each selected direction in that block's local surface basis. Fit using the resulting incidence, polar angle and azimuth. Invalid directions raise an error in the new likelihood; they are not clipped into its support.

The plane refinement initially admits endpoints within `10 nm + 0.0005 * distance_from_patch`, then iterates the final residual gate. These numerical tolerances are explicit geometric assumptions for these finite, flat-faced exported specimens. They are not a general CAD surface classifier.

Across the eligible patches, this recovers **94,692 histories labelled Transmitted**. At nominal 70 degrees, 37,354 of the 56,903 selected exits carry that label. All 466,055 selected directions lie in the local outward hemisphere. Only 26 exported exits from eligible patches fail the same-face position condition, concentrated at 60 and 70 degrees.

Sensitivity runs used final plane tolerances of 0.25, 0.5, 1 and 2 nm. Selected counts and fitted B values were identical. The corrected result is therefore not driven by the exact choice of the final plane-distance threshold within this range.

## Corrected B results

The table applies scalar-model screening comparable to the historical audit, now using the corrected population and local frames. Polar-band estimates use likelihood fits with conditional curvature errors. A pass remains a screening result, not a certificate of complete physical validity.

| Incidence | Corrected B | Current screening |
|---:|---:|---|
| 0 degrees | Unidentifiable | B has no effect at normal incidence |
| 5 degrees | 0.740210 | Provisional |
| 10 degrees | 0.601237 | Pass |
| 15 degrees | 0.692929 | Pass |
| 20 degrees | 0.676553 | Pass |
| 25 degrees | 0.700065 | Pass |
| 30 degrees | 0.725254 | Pass |
| 40 degrees | 0.835329 | Provisional |
| 45 degrees | 0.859836 | Provisional |
| 50 degrees | 0.915388 | Provisional |
| 60 degrees | 0.985839 | Provisional |
| 70 degrees | 1.038182 | Do not use as one full-hemisphere scalar |

![Historical and corrected B, with the omitted exit population](data/output/casino_surface_reanalysis_20260914/b_before_after.png)

At 25 degrees, polar-band heterogeneity falls from about 55 to 1.26. At 30 degrees it falls from about 209 to 0.76. The historical sign reversal at high tilt disappears. Much of the earlier apparent model failure was caused by how exits were selected.

The remaining high-tilt limitations should still be retained. At 70 degrees the fitted modulation approaches its positivity bound and polar bands disagree. The corrected data do not justify declaring a globally valid scalar B.

The corrected pooled candidates are:

$$B_{\rm constant}=0.971211,$$

$$B(\alpha)=0.425444+0.649093\sin\alpha.$$

The sine fit constrains the continuous-domain extrema of `B(alpha) sin(alpha)`, including any interior stationary point. Both candidates remain conditional on this material, energy, specimen geometry and selected surface population. They are not universal material constants.

## Independent checks and interpretation

The workflow fits on some repeats and tests on the omitted repeat. It also fits the constant and sine laws after omitting each whole nominal tilt. Endpoint holdouts are explicitly marked as extrapolation; interior holdouts test interpolation.

| Mean held-out NLL, lower is better | B=0 | Pooled constant | Sine law |
|---|---:|---:|---:|
| Equal-weight repeat folds | 1.663928 | 1.596748 | 1.595676 |
| Equal-weight tilt folds | 1.635355 | 1.590746 | 1.589254 |

The corrected fitted models predict held-out directions better on average than B=0. The sine law's improvement over the constant is small; average improvement does not override per-tilt adequacy failures.

A fixed 12-by-24 equal-solid-angle histogram with a half-count prior is another comparison model. Each held-out score uses a table fitted only to its training repeats. At 70 degrees its held-out NLL is 1.473561, compared with 1.651355 for a separately fitted scalar B at that tilt. This supports investigating a richer high-tilt angular response. It does not establish unseen-tilt interpolation or a ready renderer calibration.

The table stored in `angular_reference.npz` is a final all-run analysis reference. It must not be used as independent test evidence on the runs that generated it. Its cells are probability masses, not densities: divide by cell solid angle before integrating a density, or sum the appropriate masses directly.

The immediate recommendation is to retain 10–30 degrees as the expanded scalar-B calibration region for further renderer validation. No new large simulation campaign is required to obtain this improvement. The subsequent [audit repairs](AUDIT_FIXES_20260914.md) implemented the local/global detector-angle correction. Applying a fitted law still requires qualification against the renderer's actual incidence, energy and material domain.

## Running the corrected workflow

With an installed project Python environment, run from `DataSimulation`:

```powershell
python main.py benchmark-fit-b-surface `
  --cas-root data/output/casino_reruns/parametric_model_tilt_304ss_60kev_combined_0to70_fit/cas `
  --output-dir data/output/casino_surface_reanalysis_20260914

python tools/check_casino_surface_reanalysis.py `
  --output-dir data/output/casino_surface_reanalysis_20260914
```

The first command creates or validates caches, calculates local frames and surface selections, fits B, and writes held-out scores. The second recomputes tolerance checks using only the caches and plots the before/after comparison.

The project virtual-environment launcher was repaired after this surface-analysis stage. From `DataSimulation`, use `../.venv/Scripts/python.exe` instead of `python` in these examples if Python is not on PATH.

The legacy `benchmark-fit-b` command is retained for historical analysis. Its population definition differs from the corrected workflow, and invalid pooled likelihood inputs now raise an error. Use `benchmark-fit-b-surface` to reproduce the current selection and results.

## Code and evidence

- [Compact extraction and surface analysis](bse_renderer/validation/casino_compact.py)
- [Legacy parser streaming repair](bse_renderer/validation/casino_emission_fit.py)
- [Regression tests](tests/test_casino_compact.py)
- [Generated results](data/output/casino_surface_reanalysis_20260914/RESULTS.md)
- [Source hashes and cache sizes](data/output/casino_surface_reanalysis_20260914/source_manifest.json)
- [Every landing patch and selection count](data/output/casino_surface_reanalysis_20260914/surface_selection.json)
- [Per-band fits, repeat ranges and held-out scores](data/output/casino_surface_reanalysis_20260914/surface_b_report.json)
- [Tolerance sensitivity](data/output/casino_surface_reanalysis_20260914/tolerance_sensitivity.json)
- [Analysis provenance](data/output/casino_surface_reanalysis_20260914/analysis_provenance.json)
- [Test and validation record](data/output/casino_surface_reanalysis_20260914/validation_record.json)

The tests cover two different landing faces within one file, recovered globally transmitted exits, exclusion of other-face exits, retained final rows despite hundreds of intermediate collisions, forbidden whole-file reads, content-hash cache invalidation, invalid likelihood inputs, known-density B recovery, unavailable holdouts for a single run and rejection of inconsistent exported counts. At the surface-analysis stage, both suites passed 158 tests before two final guards, followed by 23 passing focused tests. After the subsequent broader audit repairs, the combined suites passed 181 tests; see the [current validation summary](docs/audit_20260914/evidence/fix_validation_summary.json). All 74 retained blocks have matching counts or the documented one-history export discrepancy, and all incident energies are 60 keV.
