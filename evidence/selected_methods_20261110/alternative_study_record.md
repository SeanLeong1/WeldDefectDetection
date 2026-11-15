# Median region-growth study

Completed a separate prospective synthetic study. Validation selected **Growth + fragment merging** from 171 new configurations. Its validation mean F1 was 0.321, compared with 0.400 for the previously selected position-and-size correction. It is the selected new candidate, not a replacement selected over every comparator. The paired intervals do not establish improvement over the original median at any tested count level. The change in mean F1 across count levels is +0.067, with a paired 95% interval [-0.103, +0.187].

Compared with the previous position-and-size correction, the mean F1 difference is -0.074 [-0.226, +0.041]. This interval does not establish a difference between the methods.

With the growth settings held fixed, enabling fragment merging changes mean F1 by +0.000, with a paired 95% interval [+0.000, +0.000].

At the same strong threshold and minimum area, enabling local growth increases mean F1 from 0.060 to 0.242, a change of +0.182 [+0.029, +0.355]. This isolates the growth step at that operating point; it does not establish superiority over the previously tuned detectors.

## What was tested?

The median background estimate, Gaussian sigma 1 pixel, training-only normalisation and per-row robust calibration stayed fixed. Strong seeds trigger lower-threshold local growth within a bounded radius. Compatible fragments can merge across a small gap if their transverse spans overlap and their combined dimensions remain at most 40 pixels (2 mm). Both contrast signs are retained as diagnostic evidence; neither sign is mandatory.

The three families were threshold only, growth, and growth plus merging. Each family was independently selected on the same 32 validation backgrounds (576 images). Training calibration comes from 96 original backgrounds. The search is finite and conditional on this renderer, known annular coordinates and pixel pitch. No CNN retraining was performed.

## What was held out?

All selections were frozen at 2026-11-10T05:44:09.747586+00:00, before generating 24 new backgrounds / 432 images. Each count level has 144 images: 48 clean and 96 pore-containing acquisitions. Those 96 acquisitions represent 48 pore geometries with two repeats, on 24 independent backgrounds. Seeds are disjoint from the original campaign and previous median follow-up. Every comparator received the same fresh images.

The original one-to-one circular box IoU >= 0.5 rule was retained. A nearby response can fail localisation. Filled-box pixel overlap is a selection tie-break for the new methods and is not contour-segmentation accuracy.

## Results

| Method | 1× F1 | 4× F1 | 64× F1 |
|---|---|---|---|
| Original median | 0.109 | 0.175 | 0.241 |
| Previous position + size | 0.250 | 0.337 | 0.361 |
| Strong threshold only | 0.110 | 0.147 | 0.250 |
| Bounded local growth | 0.246 | 0.233 | 0.247 |
| Growth + fragment merging [selected new candidate] | 0.246 | 0.233 | 0.247 |
| Frozen CNN | 0.725 | 0.868 | 0.889 |

At 64×, the validation winner correctly localised 18 of 96 pore acquisitions. Another 32 had an overlapping prediction that failed localisation; 46 had no overlapping prediction. It flagged 0 of 48 clean acquisitions.

## Interpretation and limits

The code and height maps support aligned injection: the geometric opening and depression are generated from the same cap equation. Detector differences, surface slopes, visibility, blur and sampling produce a different contrast footprint. Height is bilinearly sampled and the binary mask uses nearest-neighbour sampling, so their sampled edge support can differ.

This experiment changes region formation, not the physical renderer. Independent family selection means family-to-family results include different tuned thresholds and areas. Those family comparisons do not isolate one parameter; the predeclared matched controls isolate growth and merging at fixed operating points. Larger growth or merging can recover a pore extent but can also join noise, enlarge a box too far, or merge adjacent pores. The campaign contains one prescribed pore per defect scene, so crowded-pore separation remains untested.

Uncertainty uses 2,000 paired background bootstrap resamples, grouping repeats and noise levels. Intervals are conditional on selected methods and omit selection uncertainty. Multiple comparisons are exploratory. The previously inspected validation set is reused; fresh confirmation reduces direct test reuse but cannot establish physical transfer. Measured weld-image validation remains future work.

## Files

Open OPEN_REPORT.html for figures, operating points, counts and intervals. results.csv contains the numerical summary. Raw artifacts and frozen protocol: C:\Users\sfleo\OneDrive\CompSciEng\Thesis\Thesis\Codebase2\DataSimulation\data\output\median_region_growth_20261110_v1

Reproduce with the project Python: WeldDefectDetection/median_region_growth_20261110.py all, then WeldDefectDetection/report_median_region_growth_20261110.py. Existing completed stages are reused. All eight behavioural tests passed before the run; final file, seed, timestamp, metric and prediction audits passed.
