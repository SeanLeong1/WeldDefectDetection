# Raised-crown methods, in execution order

## 1. Geometry before pixels

The analytic fusion outline is a solid-material label. Surface height h(u,theta) is a separate boundary connected to it. The nominal crown uses H[1-(u/a)^2]^3 on its compact support, with H=3 mm and provisional a=1.4 mm. Periodic width/height variations are explicitly assumed. A 7 mm fusion depth is never assigned to a surface-trench parameter.

## 2. Pore geometry and surface expression

An open pore cuts a cap-shaped bowl into the healthy surface. An enclosed pore is represented by a void in surface-following coordinates and can have no surface effect. The displacement experiment adds a compact bump whose integral equals an assumed fraction of the pore volume. The model cannot infer that fraction from pore volume alone. Overlapping openings use the lowest floor; bulges add prescribed volumes. Visible footprint masks and individual IDs are saved separately from internal geometry.

## 3. From height to detector signals

Cartesian finite differences estimate hx and hy. The unit normal is (-hx,-hy,1)/sqrt(1+hx^2+hy^2). Detector quadrature integrates the existing angular/yield response over each quadrant. The old B fit is not extrapolated: the explicit uncalibrated Lambertian fallback sets B=0. Ray horizons reject directions blocked by another part of the height surface. This adds geometric occlusion, not electron transport. The constant local working distance and parallel-direction convention remain assumptions.

The four signals are A, B, C, D. Tx=(A+D)-(B+C), Ty=(A+B)-(C+D), and TOPO=sqrt(Tx^2+Ty^2). TOPO is a contrast magnitude, not height. A single crest can create two bright slope bands. Native raster acquisition then adds the inherited unmeasured shot/readout/drift model. No new literature-derived penetration depth is used.

## 4. Coordinates and preprocessing

Sample a complete 160x5024 annulus directly from the Cartesian image at 0.05 mm radial pitch. Score central rows 16:144, a nominal 6.4 mm band. The outer rows provide context. Training pixels alone determine the percentile-based raw range and square-root tone curve. Every sample sharing a physical background stays in one split. Ring localisation errors re-extract the native image, and scores return to fixed true coordinates before scoring.

## 5. CNN, layer by layer

Each input is one TOPO channel, shape [batch,1,64,64]. A convolution block contains two 3x3 convolutions, each followed by batch normalisation and ReLU. Spatial dimensions are preserved by padding.

| Stage | Output per example | Role |
|---|---|---|
| Encoder 1, two convolutions | 24x64x64 | Local edges and intensity patterns |
| Max pool 1 | 24x32x32 | Reduce spatial resolution |
| Encoder 2, two convolutions | 48x32x32 | Combine local patterns |
| Max pool 2 | 48x16x16 | Broader effective context |
| Bottleneck, two convolutions | 96x16x16 | Coarse contextual representation |
| Transposed convolution | 48x32x32 | Upsample |
| Concatenate encoder-2 skip | 96x32x32 | Restore detail from earlier features |
| Decoder 2, two convolutions | 48x32x32 | Fuse context and detail |
| Transposed convolution | 24x64x64 | Return to input resolution |
| Concatenate encoder-1 skip | 48x64x64 | Restore fine spatial information |
| Decoder 1, two convolutions | 24x64x64 | Produce local segmentation features |
| 1x1 output convolution | 1x64x64 | Produce one logit per pixel |
| Sigmoid | 1x64x64 | Convert logits to model scores |

Training minimises weighted binary cross-entropy plus Dice loss with AdamW and a cosine learning-rate schedule. Half the sampled patches contain visible positive truth and half are negative. Training uses labels; inference does not. Twenty epochs are fixed in advance. Validation selects the operating point and one of three trained seeds. At inference, retain central 32x32 predictions from overlapping tiles, assemble the complete ring, then threshold and extract periodic connected regions. Scores are not calibrated probabilities of an internal pore or manufacturing rejection.

For a pixel logit z, the sigmoid score is p = 1/(1+exp(-z)); its training label y is 0 or 1. Weighted cross-entropy penalises -[w*y*log(p) + (1-y)*log(1-p)], averaged over pixels. The added Dice loss is 1 - (2*sum(p*y)+1)/(sum(p)+sum(y)+1), computed over the batch. The positive weight is derived from training-patch occupancy and capped at 20. AdamW starts at learning rate 0.001 with weight decay 0.0001; cosine scheduling ends at 0.00005. Gradients are clipped to norm 5. Each epoch samples 1,200 patches, with batch size 32. The network has 262,777 parameters.

Batch normalisation uses batch statistics during training and the learned running statistics during inference. Pooling reduces spatial resolution without learned weights. Transposed convolutions learn how to upsample; skip connections preserve earlier spatial details. ReLU keeps positive activations and maps negative ones to zero. The final threshold and component rule convert scores into reported regions; they are separate from the neural-network layers.

## 6. Neighbouring median, chronologically

For each 64x64 target window, take eight spatial neighbours at angular offsets +/-64, +/-128, +/-192 and +/-256 pixels. Compute the median along each row of the target and each neighbour. Smooth their row-profile difference with the existing nine-row Gaussian and add that correction to each neighbour. Then take the pixelwise median of the eight corrected neighbours. This is intensity-profile alignment, not geometric translation of the images. Subtract that reference from the target and smooth the residual with the inherited one-pixel Gaussian. Clean training windows estimate a robust residual centre and scale separately for each radial tile position. The score is the absolute centred residual divided by that scale.

As with the CNN, keep central 32x32 scores, assemble the full ring, and apply a validation-selected threshold/component rule. Nearby defects can contaminate reference windows; smooth ridge variation and mislocalisation can create residuals without a pore. This is a spatial reference method, not a temporal prediction model. It has no learned convolutional layers.

## 7. Evaluation and what it proves

Match predicted and true circular boxes one-to-one at IoU >=0.5. Report precision, recall, F1, pixel IoU and clean-ring alarms. Bootstrap independent backgrounds, not overlapping patches. Report individual-pore recall separately from detection of a merged overlap region. Enclosed no-response controls test whether the pipeline invents a surface cue. Their lack of a simulated signal is a declared renderer assumption, not a measured SEM information-depth result.

The comparison answers whether these methods detect the declared synthetic surface anomalies and tolerate the tested coordinate errors. Measured surface profiles, detector calibration and independent real TOPO images remain necessary for physical and transfer claims.

## 8. One sample traced through the implemented pipeline

![Pilot bulge traced through geometry, detector and both CV paths](figures/raised_crown_results_20261005/traced_sample.png)

This is the declared pilot bulge, not a newly selected confirmation example. Read the panels in row order: height and inclination; four quiet quadrant signals; quiet and acquired TOPO; training-normalized TOPO and surface truth; CNN scores and final connected regions; then median reference, smoothed residual, calibrated score and final regions. The numerical arrays, eight neighbour patches and provenance are saved in `traced_sample.npz` and `traced_sample.json` in the active study. Truth is displayed to explain the sample and never enters inference.

The illustrated tile is 64 x 64 pixels. Production inference retains only its central 32 x 32 scores; final regions come from stitching the complete ring before thresholding and component filtering. The four quadrant images precede the random acquisition effects; their common focusing factor is applied when forming the saved quiet TOPO. Colourbars identify the panel scales, so displayed brightness between panels is not a shared physical scale.

## 9. Follow the code

- [Geometry](../WeldDefectDetection/raised_crown/geometry.py): `height`, `fusion_half_width`, `fusion`, `pore_contains`, `surface_effect`.
- [Rendering and native extraction](../WeldDefectDetection/raised_crown/rendering.py): `Engine.volume`, `Engine.inject`, `render`, `extract`.
- [Nonlocal detector visibility](../WeldDefectDetection/raised_crown/shadow.py): `VisibilityQuadrature.horizon`, `integrated`, `compute`.
- [Actual CNN layer definitions](../WeldDefectDetection/src/algorithms/classical/cnn_detector.py): `TinyDefectSegNet`.
- [Training, normalisation and whole-ring inference](../WeldDefectDetection/raised_crown/learning.py): `normalize`, `populations`, `batch`, `train`, `predict_image`, `median_fit`, `median_image`.
- [Median neighbours and residual](../WeldDefectDetection/circular_study/median.py): `residual`; [reference profile correction](../WeldDefectDetection/box_development/methods.py): `aligned_reference`.
- [Frozen assessment](../WeldDefectDetection/raised_crown/assessment.py): `evaluate`, `stress`, `map_back`, `estimated_error`.
- [Teaching trace](../WeldDefectDetection/raised_crown/trace_sample.py): one pilot sample through both implemented methods.
