# Chapter 4: source, equation and figure register

Current prose review: [Chapters 4–6 framing amendments](FRAMING_REVIEW_CHAPTERS_4_TO_6_20261030.md). The original equations, tables, figures and references remain unchanged.

Review date: 30 October 2026. Draft: `04_detection_methods_and_experimental_design.md`. Campaign: `count_diversity_20261027_v2`. Report preparation preserves all frozen study artifacts and excludes new training, tuning and confirmation experiments.

## Literature checks

| Chapter citation / BibTeX key | Verification on 30 October 2026 | Claim boundary |
|---|---|---|
| 1 / `roberts2017validation` | Opened the author-hosted published PDF; checked first-page bibliographic metadata and dependence-aware validation discussion. | Motivates grouped evaluation. The source concerns ecological applications and does not determine the study's sample size or certify its bootstrap coverage. |
| 2 / `nist_mad` | Opened NIST's Dataplot entry; checked the MAD definition, division by 0.6745 and normal-distribution interpretation. | Supports the scaled MAD interpretation. Does not establish Gaussian residuals or a false-alarm probability for threshold 12. |
| 3 / `ronneberger2015unet` | Opened the author preprint record; checked authors, title, date and contracting/expanding-path description. | Supports architectural motivation only. No imported pretrained weights, benchmark scores or assertion that TinyDefectSegNet reproduces U-Net. |
| 4 / `shrivastava2016ohem` | Opened the author preprint record and abstract. | Supports the motivation for selecting difficult examples. The periodic clean-only miner differs from the paper's OHEM algorithm. |
| 5 / `cawley2010selection` | Opened the JMLR record and abstract; checked authors, volume, pages and selection-overfitting claim. | Supports the need for independent evaluation after finite-sample selection. No quantitative selection-bias estimate for this experiment. |

All five keys already exist in the shared `references.bib`, which retains 22 unique entries. No new bibliography key is necessary. Chapter-local citation numbers remain temporary until thesis assembly.

## Source map and consequential distinctions

| Topic | Verified implementation or artifact | Report consequence |
|---|---|---|
| Dataset and arms | `count_study/campaign.py`: `COUNTS`, `ARMS`, `MULTIPLIERS`; current split indexes | 176 backgrounds, 3168 scenes, nested 24/96 training backgrounds. Three conditions × three counts × two repeats per background. |
| Pore-size distribution | Read one intrinsic, 1× repeat-0 row per background in each split index | Counts 26/38/32, 15/11/6, 12/17/19 across small/medium/large for train/validation/confirmation. Bin variable is the nominal parent diameter, not the smaller opening diameter. |
| Shared normalisation | `count_study/learning.py`: `prepare`; `raised_crown/learning.py`: `normalize`; saved `normalization.json` | Uses first 24 training backgrounds across all noise levels and classes. Low=0, high=1.10034207634628. The specialists do not have completely isolated count-level exposure because this transform is shared. |
| Gaussian processing | `Data.image` | Smooth raw TOPO first with radial reflection/angular wrapping; apply square root afterwards. Median searches six scales; CNN has no added input Gaussian. Do not claim a same-scale experiment. |
| Median reference | `median_residual`, `OFFSETS` | Eight samples at offsets ±64/128/192/256 columns. Every target column receives a reference, but samples are not immediate neighbours. Target exclusion concerns the reference calculation. |
| Median fitting | `median_fit`, `median_score` | Fit clean-training residuals every eighth angular column; one centre and MAD scale per radial row; floor 0.003; absolute standardised score. Clean-class metadata determine the fitting subset. |
| Architecture | `cnn_detector.py`: `ConvBlock`, `TinyDefectSegNet` | One input channel, base width 24, five two-convolution blocks, two max pools, two transposed convolutions and one output head. Direct forward hooks verify shapes; direct parameter sum gives 262777. |
| Patch populations | Current `Data.populations`, `Data.batch`; reused `tile` and `CENTRES` | 64×64 patch, four radial centres, 157 angular centres. Patches with any mask pixel enter the positive population. No random flips or rotations appear in this training path. |
| Epoch sampling | `raised_acquisition/learning.py`: `epoch_indices` | 600 positive + 600 negative draws with replacement. After mining, 300 negative draws use the hard pool. Same number of draws does not mean identical coordinates across arms with different candidate populations. |
| Current mining | `count_study/learning.py`: `clean_candidates`, `mine` | 168 candidates and 16 retained patches per independent background across applicable clean acquisitions. Do not copy the historical implementation's per-ring wording. Top-64 probability mean ranks candidates. |
| Loss | Current `train_cnn` | Weighted BCE mean plus batch-pooled soft Dice. Positive weight uses half the mean positive-candidate mask fraction. Gradient norm limit 5. |
| Optimisation and compute budget | Current `train_cnn` and `epoch_indices` | AdamW, 20 epochs, 1200 draws, minibatch 32 with final batch 16; 38 updates per epoch, 760 total, 24000 draws. Cosine scheduler steps after each epoch; saved candidates at epochs 10 and 20. CPU, four PyTorch threads. |
| Inference | `raised_crown/learning.py`: `predict_image` | 628 input patches per ring; central 32×32 outputs tile 128×5024 without averaging. Circular context comes from wrapped extraction, not circular convolution padding inside the network. |
| Selection | Current `select`, `objective`, `selection_key`, `develop` | Mean per-count F1, then pooled pixel IoU, then fewer FP. Current area grid 3/8/15 differs from the older metrics module's generic selector. Current selection imposes no separate low-false-alarm constraint. |
| Morphology / matching | Current `component_rows`; `circular_study/metrics.py`: `regions`, `binary`, `circular_iou`, `match` | Threshold then closing then area cutoff. Periodic 8-connectivity. Assignment maximises valid pairs first and summed IoU second. Truth uses minimum area 1, not the prediction cutoff. |
| Aggregation / uncertainty | `metrics.summary`, `metrics.bootstrap`; current `result_summary`; `reporting.paired` | Pooled region counts and pooled pixel IoU, clean-acquisition alarm denominator. Background bootstrap uses 2000 replicates. Paired differences reuse sample indices, with fixed model selections. |
| Frozen boundary | `frozen_models.json`; execution record; independent final audit | Freeze 13:33:43 UTC; earliest confirmation raw file 13:35:40 UTC, 27 October 2026. v1 retired before training following repeat-seed overlap. No v1 result enters this chapter. |
| Retained controls | Current `evaluate`; earlier acquisition campaign identity | Two old models retain their normalisation/settings. Current experiment evaluates 18 CNN seeds + 6 medians + 2 controls = 26 summaries, each over 864 confirmation scenes. |
| Diagnostic exclusion | `reporting.diagnostic`, `diagnostic_evaluation` | Single fixed background, nominal pore diameter 0.2 mm, noiseless/shot-only/full acquisition; excludes fitting, selection and principal confirmation. |

## Equation checks

| Equation | Check |
|---|---|
| 4.1 | Quantile bounds and 8% padding match `prepare`; clipping precedes square root. |
| 4.2 | Gaussian applies to raw rather than normalised intensities; sigma is standard deviation, not variance. |
| 4.3 | NumPy rolling yields the same symmetric offset set. For eight values, NumPy averages the fourth and fifth sorted values. |
| 4.4–4.5 | Centre and scale pool sampled columns from clean training only; floor and absolute residual match source. |
| 4.6 | BCE uses a positive-class weight, default mean reduction and logits internally. Dice sums all batch pixels. |
| 4.7 | Pixel prevalence estimate matches the exact half-weighted positive-candidate formula. |
| 4.8 | Selection uses mean per-count F1, not pooled F1, and shares the same tie-break order across methods. |
| 4.9–4.10 | Box IoU ≥0.5 controls matching. Region counts precede precision/recall/F1 calculation. Zero denominators return zero. |
| 4.11–4.12 | Pixel IoU pools intersections/unions; clean alarm counts acquisitions with any prediction. Repeated variants remain dependent. |
| 4.13 | Paired bootstrap subtracts F1 computed from the same resampled background groups. No retraining or reselection inside a replicate. |

## Figure provenance

Run `.venv\Scripts\python.exe docs/thesis_draft_20261030/make_detection_figures.py` from the repository root. The script checks generation and development source hashes, checks selected raw/geometry hashes and records its source identity and figure hashes in `evidence/chapter4_provenance.json`. It writes only report artifacts. PNG, PDF and SVG exports accompany each figure.

| Figure | Input and selection | Boundary |
|---|---|---|
| 4.1 | Protocol background/scene counts and split roles | Explanatory diagram, not a measured result chart. Training figure shows the whole split; individual arms restrict it as Table 4.2 specifies. |
| 4.2 | First training background, clean and intrinsic, 1× repeat 0; frozen `mixed_96` median | Recomputes the full-ring median and region mask, then crops columns −32…32 modulo 5024 and original radial rows 48…111. No performance-based sample selection. This is a training example, not evidence of generalisation. |
| 4.3 | Actual class forward pass with input shape 1×1×64×64, block hooks and parameter count | Architecture diagram. Dummy values verify dimensions without loading, training or selecting a checkpoint. |

Figure 4.2 displays common scales within each column; the score display clips at 24 only for presentation. The full clean ring contains three predicted regions although its local display crop contains none. The full pore ring contains five predicted regions; the crop shows only part of that output. The chapter caption preserves this distinction and uses a green dashed geometry contour rather than presenting the contour as a predicted mask.

The network figure's diagonal arrow initially crossed its upsampling label. The revision moves the label and preserves the actual skip/decoder paths. Visual inspection covers each final PNG, including text, shape labels, colour scales and contour placement.

## Editorial and assembly checks

The chapter follows dataset design, shared input, median, CNN, selection and evaluation. It contains 13 numbered equations, four tables and three figures. It keeps principal performance scores for Chapter 5 and does not retrospectively describe the implemented CNN as a same-Gaussian-scale comparator.

Authored prose uses active voice and avoids first-person pronouns. Literature attribution distinguishes architectural motivation and validation principles from source-verified implementation details. Existing chapter/section/equation references and local links require checks alongside the citation keys; Chapters 5 and 6 remain planned forward references. Final LaTeX assembly still needs labels, unified citation numbering and a clean compilation.
