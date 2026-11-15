# Chapter 5: current evidence and figure register

Review date: 11 November 2026. This register supersedes the 30 October chapter-specific review. The historical receipts remain available for the unchanged noise figure and earlier draft.

## Scope and evidence

The revised results chapter presents the final selected CNN and Median, CNN training-design findings, localisation failures and the CASINO response check. It removes superseded Median settings, old-model transfer tables, detailed tuning history, repeated caveats and obsolete workflow timings. It uses question-led sections, active subjects and a completed-research perspective.

The final detector comparison descriptively pools two fresh 24-background confirmation sets. Identical CNN and Median settings precede both sets. The original 48-background confirmation population remains separate and supports the CNN training-noise, diversity and seed comparisons. The chapter calls the final classical method Median throughout; its method description explicitly retains the positional shift and box padding.

| Result | Current source | Check |
|---|---|---|
| Table 5.1, population accounting | Original training-study audit; both fresh-set audits | 3168 training-design images, 96/32/48 backgrounds; 864 additional final-confirmation images on 48 backgrounds |
| Equation 5.1, Figure 5.1 | Saved measured noise and original chapter provenance | Existing 48-background repeat-variability evidence; unchanged figure |
| Tables 5.2–5.3, Figure 5.2 | Final selected-method summary and four per-scene result files | Re-summed six method/count groups; checked TP/FP/FN, overlap totals and F1; verified source hashes |
| Sections 5.3–5.4, Table 5.4, Figure 5.3 | Original all-seed results and paired comparisons | Retained validation choices and original test population; checked rounded training scores |
| Section 5.5.1 | Final pooled localisation summary; geometric alignment audit | 68 correctly localised, 79 overlapping but insufficient, 45 without overlapping predictions at 64x |
| Table 5.5 | Per-scene results from both final confirmation sets | Recomputed low-noise size-group counts and recall for both methods |
| Figure 5.4 | Saved final-set-A TOPO arrays, geometry masks and predicted boxes | Backgrounds 000 and 012, intrinsic pores, 64x, repeat 0; recomputed best circular box IoU |
| Table 5.6 and transport interpretation | Reproduced angular report and polar discrepancies | Retained CASINO population; fold-specific training estimates; B=0 remains the imaging assumption |
| Software outcome | Current GUI documentation and source | Two workspaces, stage previews, folder processing and optional mask statistics; no new performance claim |

## Reproduction

Run `refresh_selected_results.py` from this directory using the project Python environment. The script reads saved evidence, checks source hashes, aggregates current counts and writes three PNG/PDF figure pairs plus `evidence/chapter5_refresh_20261111.json`. It performs no training, rendering, selection or new confirmation experiment. The historical `make_results_figures.py` reproduces the earlier figures and is not the current final-detector report generator.

Visual inspection covered all three new figures. Labels and numerical annotations remain legible; error bars and common scales are explicit. Failure panels show the actual image, geometric opening and predicted boxes. The small-pore example has an unmatched Median prediction, so the caption distinguishes a localisation miss from the CNN's absence of a surviving region.

The revised chapter adds no literature claims or bibliography entries. Project evidence labels P1–P8 resolve to local sources. The verification receipt records source and figure hashes. Small dependent edits in Chapters 4 and 6 identify the final method, update the conclusion counts and remove obsolete cross-references or completed software tasks from future work.

## Assembly boundary

This request updates the Markdown draft in `thesis_draft_20261030`. It does not regenerate or overwrite the separately edited LaTeX project. The old `assemble_overleaf.py` writes to that project and must not run blindly: its historical assumptions and manual-number mappings require review for the new section and figure structure. Markdown validation does not establish a compiled PDF.
