# Raster acquisition model v2, 28 September 2026

This is a mechanistically structured acquisition surrogate with **unmeasured amplitudes**. It replaces the historical arbitrary alternating detector row/column drift in new GUI previews and generic batches. `ScanEngine` retains an explicit legacy mode for historical reproduction. Existing frozen datasets and models are preserved. A separate v2 campaign retrains and evaluates the CNN and neighbour-median method.

## Physical interpretation and equations

The procedural surface, constant-B angular model, detector integration and interaction blur first produce four nonnegative quadrant signals `q_A..q_D`. These are simulation intensity units, not measured electron counts. CASINO evidence supports parts of the angular model, not this entire image-generation chain.

1. **One raster clock.** For row i, column j, width W, dwell d and flyback f in seconds, `t(i,j)=i(Wd+f)+jd`. All four detectors use this clock. No detector is assigned an unrelated vertical scan direction.
2. **Separate geometric motion.** Optional constant velocities map the image to coordinates `(x-v_x*t/s, y-v_y*t/s)`, where s is mm/pixel. Quadrants and labels use the same map; intensities use bilinear interpolation and masks use nearest-neighbour interpolation. Border samples use nearest-edge extension. The current campaign uses zero velocity. This supports displacement testing, not a model of changing beam incidence or shadowing.
3. **Slow contamination surrogate.** Optional common attenuation is `exp(-k*t)`. Fixed channel gains remain explicit. The small k range is assumed; this is not a measured contamination growth law.
4. **Electronic gain and offset.** Common multiplicative gain is `max(0,1+sigma_g*u(t))`; each detector has an independent additive offset `sigma_b*u_k(t)`. The stationary Ornstein-Uhlenbeck recurrence is `u[n+1]=a*u[n]+sqrt(1-a^2)*epsilon[n]`, `a=exp(-Delta_t/tau)`, initialized with `u[0]~N(0,1)`. Thus node variance is one instead of growing with image dimensions. Slow fields are sampled at line boundaries and linearly interpolated within lines. This approximation attenuates intra-line variance when line time is not small relative to tau; metadata records their ratio. It is not an exact pixel-resolved OU process.
5. **Counting and readout.** With effective conversion C, `N_k~Poisson(q_k*C*d)` and measured counting signal is `N_k/(C*d)`. Hence variance is `q_k/(C*d)`. Independent readout has RMS `sigma_r*sqrt(d_ref/d)`, an assumed white-noise averaging model. C is an effective unknown count conversion, not a detector efficiency or an asserted primary-electron count. Beam-current effects also enter the preceding signal gain; C is not separately fitted to current. Signals are finally clipped nonnegative, which can bias very weak signals.
6. **Channel formation.** `Tx=(A+D)-(B+C)`, `Ty=(A+B)-(C+D)` and the implemented `TOPO=sqrt(Tx^2+Ty^2)`. The magnitude has a positive noise floor. Even zero-mean detector errors do not produce a zero-mean TOPO error. Common gain on identical flat channels cancels in Tx/Ty, whereas unequal offsets and independent counting/readout noise do not.

Geometry keeps its original seeded random stream. Acquisition uses separately named deterministic SeedSequence streams for profile sampling, common gain, channel offsets, channel readout and channel counting. Clean/pore pairs share electronic fields. Poisson draws depend on their input signal, so identical seeds do not imply identical shot-noise pixels after geometry changes.

## Assumptions frozen before training

| Parameter | Reference | Development and confirmation sensitivity range | Units |
|---|---:|---:|---|
| Effective count conversion C | 1e9 | 5e8 to 2e9, log-uniform | counts / (raw quadrant unit second) |
| Readout RMS at 8 us | 0.00045 | 0.0002 to 0.0008 | raw quadrant units |
| Independent baseline RMS | 0.00020 | 0.0001 to 0.0004 | raw quadrant units |
| Common gain RMS | 0.002 | 0.001 to 0.004 | fraction |
| Correlation time tau | 0.25 | 0.1 to 1.0 | seconds |
| Flyback | 20 | fixed 20 | microseconds |
| Attenuation rate k | 0.0002 | 0.0001 to 0.0004 | inverse seconds |
| Geometric velocity | 0, 0 | fixed 0, 0 | mm/second |

Other ranges are uniform and sampled independently per latent background. They are numerical sensitivity ranges, **not empirical distributions of good or bad acquisitions**. Lower F1 or higher F1 cannot validate the chosen noise amplitude. The legacy temperature, bandwidth, contamination-rate and cumulative-session parameters are intentionally unused by v2 and marked as such in metadata and the GUI. V2 currently starts its acquisition clock anew for each image; inter-image drift continuity is not modelled.

## User-supplied reference

The screenshot supplied on 28 September contains two displayed circular specimens with multiple experimental rings, yellow crosshairs, UI elements and strong directional contrast in the right panel. The user identifies these as poor BSE examples. The channel names, raw quadrant data, physical pixel size, dwell time, detector gains and display transfer are unknown. Therefore it is retained as qualitative context only. Neither panel is used as a noise calibration target, CV ground truth or proof of defect detectability. Ring count is specimen/test history, not a reason to inject rings into every synthetic weld.

The horizontal discontinuities in this screenshot could involve scanning, display/compositing or specimen contrast. Their cause cannot be determined from the screenshot alone. Our unsigned TOPO magnitude is not necessarily equivalent to either displayed instrument channel.

For measured calibration, obtain repeated raw frames of one stable region at known dwell/current, raw channel identity or all four quadrants, mm/pixel, scan direction and gain/display settings. Register frames to estimate geometric displacement separately. Use temporal residuals on stable regions for count/readout statistics and channel covariance, then fit offset autocorrelation in scan time. Exclude crosshairs/UI/annotations and test-ring edges. Keep a separate held-out specimen for CV transfer with expert annotations.

## New experiment

`WeldDefectDetection/run_acquisition_campaign_20260928.py` freezes sources and this protocol before data generation. Development has 100 backgrounds, each with clean/keyhole/intrinsic variants: 60 train, 20 validation, 20 reserved test. Smooth width/relief variability cycles through 0, 5 and 10 percent, separately from beam-regime cycling. Train-only normalization operates on raw float32 data. Three CNN seeds run 20 epochs; validation chooses checkpoint and postprocessing. The median method uses four distinct neighbours excluding the target. No arbitrary folder is assumed to supply such context.

Models are frozen before 30 new confirmation backgrounds, each rendered at all three variability levels with all three classes. This produces 270 confirmation scenes / 1,350 target windows. Results include noisy and same-geometry quiet images, per-pore outcomes, false alarms and background-cluster bootstrap intervals. Quiet testing is a diagnostic distribution change, not a separately selected model. An 80 mm circular-weld example is a transfer illustration, not a statistically independent validation population.

The screenshot does not change the physical scope: the generator models surface-expressed relief and pores. An 8–10 mm process keyhole depth does not imply BSE can image buried defects at that depth. Physical acceptance of welds and transfer to genuine BSE remain unvalidated.

## Primary references for the mechanisms, not the numerical settings

- [NIST: establishing visibility in SEM](https://www.nist.gov/publications/high-resolution-often-sought-sem-imaging-but-establishing-visibility-may-be-challenge), electron statistics and visibility.
- [NIST: real-time scanning drift correction](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=903741), geometric drift is an image-coordinate issue.
- [JEOL SEM glossary: backscattered electron image](https://www.jeol.com/words/semterms/20121024.014159.php), detector-dependent BSE contrast.

These sources do not establish the v2 profile amplitudes, constant-B universality, or similarity to the supplied specimen.
