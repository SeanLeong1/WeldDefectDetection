# Chapter 3: source, equation and figure register

Current prose review: [reader-focused amendments to Chapters 1–3](FRAMING_REVIEW_CHAPTERS_1_TO_3_20261030.md). The revised explanation retains all displayed equations, table rows, references and figure artwork. Existing replay evidence remains the numerical basis for the chapter.

Review date: 30 October 2026. Chapter: `03_synthetic_imaging_methodology.md`. This chapter documents the existing `count_diversity_20261027_v2` campaign. Chapter preparation changes only report artifacts and a local figure-replay cache; it does not change generation sources, datasets, training or selection.

## Literature and citation boundaries

| Chapter citation | Key | Verification and permitted claim |
|---|---|---|
| 1 | `drouin2007casino` | Reuses the verified PubMed metadata and abstract from the Chapter 2 register. Supports the description of CASINO as an electron-microscopy modelling tool. No newly asserted full-text result. |
| 2 | `demers2011casino` | Reuses the verified PubMed metadata and abstract from the Chapter 2 register. Supports the distinction between electron-transport simulation and the local empirical renderer. No calibration claim. |
| 3 | `scipy_pchip` | Opened the official SciPy `PchipInterpolator` documentation on 30 October 2026. Checked shape preservation, continuous first derivative, monotonicity and non-overshoot. The chapter does not claim continuous second derivatives. Documentation describes the method, not the installed library version or a physical weld law. |

The shared `references.bib` now contains 22 unique entries. Chapter 3 adds only `scipy_pchip`; CASINO entries already exist. Keep the older `introduction_references.bib` separate to avoid duplicate keys. Chapter-local numbering remains temporary until LaTeX assembly.

## Equation and implementation map

| Equation / topic | Exact source and check |
|---|---|
| 3.1–3.3, circular crown | `raised_crown/geometry.py`: `Geometry`, `height`; `raised_crown/rendering.py`: `Engine.volume`. Analytic compact polynomial defines the exposed crown. PCHIP applies only to the subsurface fusion outline. Native image size equals 1921, not 1920. |
| Fusion anchors | `fusion_half_width` and `fusion`: depths 0/5/7 mm, half-widths 1.4/0.6/0 mm. Single-surface rendering does not perform volumetric transport through this solid outline. |
| 3.4–3.6, pores and masks | `pore_field`, `surface_effect`, `pore_contains`; `count_study/campaign.py`: `pore_plan`. The open cap uses the 0.35 offset. Maximum depression equals 0.65 times `depth_mm`. Opening diameters use the factor sqrt(1 − 0.35²). No new fitted pore model. |
| Volume distinction | Full analytic pore volume and discrete surface-volume change have different domains. Neither constitutes a measured pore volume. The active main dataset excludes hidden-pore and bulge counterfactuals. |
| 3.7, normals | `bse_renderer/renderer/scan_engine.py`: `_normals_from_heightmap`; native finite differences use physical spacing. Equation omits the small numerical denominator stabiliser explicitly. |
| 3.8–3.9, intensity law | `raised_crown/shadow.py`: `integrated`; `physics/emission_model.py`; `physics/bse_yield.py`; scan-engine gain calculation. B=0 reduces the emission term to the clipped local direction cosine. The 0.58/0.42 mixture, exponent 1.76 and yield polynomial remain empirical code choices. |
| 3.10–3.11, quadrature and visibility | `renderer/quadrant_detector.py`; `raised_crown/shadow.py`: `VisibilityQuadrature`. Four 90-degree sectors, 6 by 8 angular integration; horizon grid 0.1 mm, angular stride 4, ray step 0.1 mm. No claim of exact continuous-ray visibility. |
| 3.12, instrument blur | Scan-engine `_interaction_blur_px`. Formula-only mode supplies unit calibration scales; the geometry response overrides also return ones. Distinct from CV Gaussian smoothing. |
| 3.13–3.15, acquisition | `noise/raster_acquisition.py`: `pixel_times`, `stationary_ou`, `electronic_field`, `acquire`; `run_acquisition_campaign_20260928.py`: `profile_for`. OU stationarity applies to line-boundary samples; interpolation need not preserve exact unit variance at every intermediate pixel. |
| Count intervention and Table 3.1 | `count_study/campaign.py`: `dwell`, `background`, `pore_plan`; inherited `profile_for`. Only effective count rate changes within each matched count series. Two repeats use seed + repeat × 100000000. |
| 3.16, TOPO | `renderer/image_formation.py`: `form_channels`; `raised_crown/rendering.py`: `render`. Focus gain multiplies the TOPO magnitude. The displayed Tx and Ty definitions denote the differences before that gain; the scan engine also scales its exported directional channels. The count-study files contain only raw magnitude. |
| 3.17, extraction | `raised_crown/rendering.py`: `extract`, `ROWS`, `COLS`, `SCORE`. Bilinear intensities/heights and nearest-neighbour masks. Centreline pitch ≈0.05002536 mm; radial pitch 0.05 mm. |
| Table 3.2 | Direct transcription and rounding of `noise_checks.json`; `noise_check` defines variance and bias assertions. The check uses a constant field with electronics and contamination disabled. |
| Dataset audit | `independent_final_audit.json`: 96/32/48 backgrounds, 1728/576/864 scenes, 3696 unique data-file hashes, four-pixel minimum target, disjoint background/acquisition seeds and matched variants. This chapter reports the existing audit rather than rerunning CV evaluation. |

All paths in the table refer to modules beneath `WeldDefectDetection` or `DataSimulation` as linked in the chapter's P1–P8 register. Those local evidence links support implementation claims; they do not substitute for an external source about physical accuracy.

## Figure provenance and reproduction

`make_methodology_figures.py` verifies all 154 generation-source hashes against the frozen protocol and the raw/geometry hashes for the 18 scenes of `train-000`. The first run reconstructed its sample parameters and rendered the clean example followed by the intrinsic example, preserving the original cache order. Assertions established bit-for-bit equality for height, mask, quiet TOPO and acquired 1× repeat-0 TOPO in both cases.

The script saves intermediate quiet quadrants to `evidence/chapter3_train000_trace.npz`. Later runs verify the cache hash and source/input hashes before making the figures. The provenance JSON records the input hashes, replay checks, source count, generator hash, figure hashes and pore parameters. All outputs stay inside the report directory.

Reproduction from the repository root uses `.venv\Scripts\python.exe docs/thesis_draft_20261030/make_methodology_figures.py`. A missing trace cache triggers replay; an existing verified cache avoids repeating the renderer. Each figure has PNG, vector-container PDF and SVG exports; Matplotlib rasterises the 3D surface inside Figure 3.1's vector exports.

| Figure | Content and selection | Interpretation |
|---|---|---|
| 3.1 | Equations for the nominal section/fusion outline; train-000 surface; common phases with a=0/0.05/0.10 | Formula illustration. The variation comparison does not add dataset samples. Vertical exaggeration highlights the ring crown. |
| 3.2 | Saved height and masks for train-000 clean/intrinsic/keyhole | First training index, without performance-based selection. Shared scales preserve height comparisons. Wrapped columns −25…25 and rows 60…99 centre the window on the seam pore. |
| 3.3 | Replayed quiet intrinsic quadrants; saved quiet and 1×/4×/64× repeat-0 TOPO | Shared intensity scale within each row. Red contours identify geometry truth only. The panels do not show CV predictions or establish average detection performance. |

The example uses D=0.49386496514620404 mm and d=0.1867263019074587 mm. The analytic maximum depression equals 0.12137209623984816 mm. The saved unwrapped maximum equals 0.11943817138671875 mm. The receipt labels the latter `example_sampled_max_depth_mm` because interpolation precedes this measurement.

## Visual and editorial review

- Inspected all three PNG figures. Checked axes, units, shared scales, mask placement and correspondence with the text. Removed an overlapping vertical-axis label from Figure 3.1 and inspected the revised version.
- Reviewed all authored chapter prose, captions and table entries for first-person pronouns, passive constructions and ambiguous actors. Source titles retain exact published wording.
- Kept the physical depth distinctions and exact TOPO definitions beside their relevant equations. The chapter does not equate the 7 mm fusion depth with imaging sensitivity, or effective counts with primary electrons.
- Checked local file links, citation keys and references to equations, figures and tables. Chapter 4 remains an explicit forward reference to planned work.
- Markdown review does not establish a clean LaTeX build. Final assembly must replace temporary evidence labels, assign cross-reference labels, compile and resolve citation/reference warnings.
