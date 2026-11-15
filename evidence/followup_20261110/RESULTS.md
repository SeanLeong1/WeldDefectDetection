# Angular model: results and next steps, 15 September 2026

Status: experimental model implemented and frozen for evidence collection. Production qualification remains incomplete.

## 1. What the model represents

The angular model describes the directions of electrons that leave the selected CASINO entrance surface. The polar angle measures departure from its normal; azimuth measures direction around that normal. Integrating the probability density across a detector aperture predicts the fraction of outgoing electrons reaching that detector. Yield, empirical image gain, blur and defect response are separate parts of image formation.

The B laws keep the polar marginal fixed at p(mu)=2 mu, where mu=cos(polar angle). Both constant B and sine-dependent B therefore require E[mu squared]=0.5. The new model can change that marginal and the azimuthal pattern.

## 2. What was changed and tested

The analysis verified 84 source runs, 42,000 incident histories and 13,670 selected exits. It reused compact boundary caches and preserved the surface-selection and source-hash checks.

Instead of fitting 288 largely independent cell counts per tilt, the new model shares a small set of smooth coefficients across cells and tilts. Exponential corrections to Lambertian bin masses preserve positivity; normalization forces their sum to one. Ridge regularization discourages coefficients that follow sampling noise. The [pre-fit protocol](protocol.md) specifies the exact formula, features and bounded search.

Selection chose 8 coefficients with ridge 0.01. Repeats 1 and 2 fit the coefficients, repeat 3 chooses the recipe, and repeat 4 checks it. No further recipe was tried after examining repeat 4. All these files were examined earlier in the project, so none is fresh confirmation.

## 3. Results from saved artifacts

Positive gain means lower negative log likelihood, in nats per exit, averaged with equal file weight. These are descriptive file-level t intervals, not qualification certificates.

| Comparison | Mean gain | 95% interval |
|---|---:|---|
| Development repeat 4: new model over matched constant B | 0.003630 | [-0.002597, 0.009856] |
| Development repeat 4: new model over Lambertian | 0.028288 | [0.007579, 0.048997] |
| Intermediate-tilt diagnostic: new model over constant B | 0.005072 | [0.000347, 0.009796] |
| Intermediate-tilt diagnostic: new model over old half-count table | 0.091417 | [0.082383, 0.100451] |

Source: [report.json](report.json), with individual scores and split details. The intermediate-tilt check fits the selected recipe to original selection knots and scores the intermediate knots. It is a development diagnostic because the recipe was selected using the combined development campaign.

At 60 degrees, the new model predicts E[mu squared]=0.449783; the check run measures 0.408737, versus the B law's fixed 0.500000. This improves the polar statistic but does not eliminate the discrepancy.

The selected-aperture detector check has 0/1008 simultaneous conditional intervals entirely within +/-0.02. The largest point error is 0.080253; the largest difference between the two integration grids is 0.003852. These intervals assume independent histories, which repeat labels alone do not prove. They test three chosen pilot geometries, not the complete production aperture family. See [detector_development_check.json](detector_development_check.json).

The original half-count table used all four repeats at the original selection knots. Its scores on those same knots are partly in-sample. The report records them for context, but only the matched constant and Lambertian baselines enter the development advancement decision.

## 4. Decision

The new model has a positive average gain over constant B on repeat 4, but the interval includes zero. The predefined advancement condition did not pass. No all-development refit was promoted and no production qualification was issued. The repeat-1/2 experimental model and its matching constant-B comparator are frozen for new evidence collection.

Experimental artifact SHA-256: `958e92cb193c2212ee4b2caf6f5dbe673823e8a55709e121b8af86a82dc0e580`. [experimental_freeze.json](experimental_freeze.json) binds its identity; [experimental_candidate_model.json](experimental_candidate_model.json) references source evidence and archived code.

## 5. Run the next fixed validation batch

The 84 existing confirmation files are untouched, with four repeats at each of 21 tilts. Each remains at 500 incident histories. This is an initial independent batch, not an assurance that the tight detector-accuracy gate can pass. Changing the number of histories requires a separately prepared manifest and updated hashes; do not edit these inputs after launching.

From the Codebase2 root, run:

```powershell
& '.\DataSimulation\tools\casino\run_angular_confirmation.ps1' -MaxParallel 4
```

This verifies the frozen model and its evidence, then runs at most four CASINO jobs. Export each completed CAS to its matching *_summary.dat with the existing export workflow. Repeat the same command to verify and compact those files before the next wave. The model hash is pinned at the first actual confirmation launch. No simulations were started by this development analysis.

Complete the fixed batch before examining its model scores. Then run:

```powershell
.\.venv\Scripts\python.exe .\DataSimulation\tools\casino\check_angular_confirmation.py `
  --root .\DataSimulation\data\input\casino_qualification_20260914 `
  --freeze .\DataSimulation\data\output\angular_model_development_20260915_v3\experimental_freeze.json `
  --output .\DataSimulation\data\output\angular_confirmation_20260915
```

The checker rejects incomplete batches, changed models, changed sources and duplicate boundary histories. It scores the frozen models without refitting. This command produces likelihood and polar summaries; it does not issue a release certificate.

## 6. What still blocks production

- Fresh full angular-adequacy and detector-acceptance evidence.
- Evidence covering local effective-material modifiers, rather than only one simulated alloy.
- Geometry convergence and independent RNG provenance.
- A passed emission release gate before production datasets, training and final detection evaluation.

The renderer can load the experimental table, but its production qualification guard correctly rejects it. A software test pass does not replace the physical evidence.

Reproducibility: the archived protocol and source snapshot accompany this report. The first execution failed only while serializing a NumPy integer in detector diagnostics. The second completed analysis was preserved. Version 3 adds explicit baseline-overlap labels and the experimental freeze, with identical recipe, development-check scores and tilt-diagnostic scores. No additional candidate search was performed.
