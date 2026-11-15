# Weld Vision: create and inspect

Open `Launch Weld Vision.cmd`. The default now has two separate workspaces: **Create welds** and **Inspect images**. Alt+1 and Alt+2 switch between them. Tab moves through controls, and settings scroll into view as they receive keyboard focus. Both settings panels also support mouse-wheel scrolling. Run actions and progress stay outside the scrolling panels.

## Create welds

Set crown height, width and profile variation; choose a clean weld, round pore or elongated pore. Manual pores expose diameter, depth parameter, angle and radial position. Acquisition exposes dwell and 1x/4x/64x effective counts. The ring diameter remains 80 mm and sampling remains 0.05 mm. The response remains B=0.

Choose the starting seed and number of welds (1–100). Successive seeds vary backgrounds reproducibly. Optional pore variation draws diameter, depth and position from the study's pore plan instead of the manual pore values. This is independent demo generation, outside research splits. Changed crown dimensions are exploratory relative to the pretrained methods' 3 mm high, 2.8 mm wide training crown.

**Generate welds** saves each image and its actual intermediate arrays. Use the **Stage** selector for the 3D surface, healthy profile, injected pore and mask, four detector channels, directional signals and TOPO. Pore views include a magnified neighbourhood. The 3D height axis is enlarged for readability. TOPO is signal contrast magnitude, not height. Channels are formed on the native Cartesian raster before annular interpolation.

Each weld folder contains `topo.npy`, `topo_mask.npy`, `stages.npz`, `scene.json` and a `previews` folder of PNG figures. Outputs are under `data/final_method_inspection/generated_batches`. **Inspect generated images** transfers the completed set to the inspection tab and enables evaluation against its geometric masks. **Stop after current weld** preserves completed outputs.

**Load saved welds** reopens those stage arrays after restarting the app. Parameters in the left panel apply to the next generation; each saved weld retains its own settings in `scene.json`.

## Inspect images

Choose a single image or an entire folder, optionally including subfolders. **Run inspection** applies both final methods. **Image preview** shows the selected result; **Folder statistics** shows one row per input, candidates for each method, processing time and any input or reference-mask problems. Select a result row to load its preview. The run summary includes candidate totals and images flagged by each method. A failed input is reported and the run continues. **Stop after current image** saves a partial summary.

Raster batches require **Prepare raster batch** and confirmation that images share the same alignment. The first raster supplies the registration settings; images with different dimensions are reported as errors. Images with different centres, radii or orientation require individual preparation, even when their dimensions match. Multi-frame TIFF batches use the selected frame from each file, not every frame.

Optional reference masks are named `<image_stem>_mask.npy`, beside the input or in a chosen mask folder with matching relative subfolders. Masks must be binary, with shape 160 x 5024 (including halo) or 128 x 5024 (scored region), already aligned to the prepared annulus. Cartesian masks require preparation separately. Mask lookup and scoring occur after inference. Missing or invalid masks do not stop detection; accuracy uses only the successfully labelled subset, whose size is shown. Precision, recall, F1, detected pores, false positives and misses use one-to-one periodic box IoU >= 0.5, with connected components from the reference mask. Undefined ratios display N/A. Candidate totals alone are not accuracy.

Folder runs save `summary.json`, `images.csv` and per-image prediction arrays, metadata and visualisations under `data/final_method_inspection/batch_results`. The CSV includes TP/FP/FN when reference masks are available. Inference seconds exclude file import and export; the first inference includes checkpoint loading. Known auxiliary masks, stage archives, preview folders and prediction exports are excluded from folder discovery.

- **Try saved example** runs both final detectors on the saved low-noise example.
- **Open image** accepts TIFF, PNG, JPEG, NPY and NPZ. PDF is excluded. Raw arrays use a 160 x 5024 unwrapped annulus, including the radial halo, with a 40 mm ring radius and 0.05 mm radial sampling. Click **Inspect weld** to run both methods.
- Raster images open **Prepare image**. Set the centre and centreline radius, angle direction and intensity interpretation, then check the annular preview. Click the original image to move the centre. The orange boundaries show the required full 8 mm radial region. Multi-frame TIFFs support frame selection; native grayscale values, including 16-bit and floating-point TIFFs, are retained before preparation. Colour images use recorded luminance conversion. Transparent or out-of-bounds annuli are rejected.
- **Display image** applies an explicitly exploratory percentile mapping into the frozen normalization range. This does not recover raw detector values. **Raw TOPO units** preserves the declared units with the entered gain and offset. Neither path establishes physical validation. For pre-unwrapped raster images the dimensions must already be 160 x 5024; radial row direction can be reversed explicitly.
- Prepared inputs and their provenance are saved under `data/final_method_inspection/imports`. Original files are preserved. External-image exports record source hashes, frame, registration, intensity transform and exploratory status. Circular-image detections are mapped back onto the original image when it remains available and unchanged.
- Switch the **CNN / Median / Both** view without rerunning inference. The view remains centred on the same CNN-selected region for comparison.
- **Open saved results** opens the current run folder with its full-ring views, numerical score maps, masks and prediction records.

The CNN is the frozen `mixed_96` seed 1027001, epoch 20 checkpoint. Its operating point is threshold 0.9, minimum area 8 and closing 3. Median uses Gaussian sigma 1, threshold 16, minimum area 8 and closing 3, followed by the selected 2-pixel crown-directed shift and 1-pixel box padding. Its output masks are filled corrected rectangles, matching the final evaluation.

Inference uses the original normalization and clean-training calibration. It never reads reference masks. Candidate counts are predictions, not true-positive counts; ground-truth labels are required to calculate accuracy. Imported ring registration is user-specified; automatic ring localisation remains future work. The frozen detector assumes an 80 mm diameter weld. A photograph or height map is not TOPO contrast.

Older generator and inspection workflows remain under **Research tools > Open historical research tools**, or `launch_weld_studio.py --research`. Their model settings are unchanged. The two-workspace studio is the default route to the thesis-selected methods.

