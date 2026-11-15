# Submission validation

Checks completed on the packaged copy on 15 November 2026, using the existing Python environment recorded in `requirements-submission-environment.txt`:

- All package file hashes verified before prediction replay.
- Both the frozen CNN and corrected median reproduced the original saved prediction boxes exactly on the bundled pore example and clean example.
- The command-line demo ran from a different working directory and saved its figure, score maps and numerical evaluation. The bundled pore example gave one true positive, zero false positives and zero misses for each detector. This is a single illustrative example, not a new aggregate performance estimate.
- Three targeted existing regression tests passed: circular-component/matching parity, paired count-level geometry/dwell preservation, and acquisition-seed disjointness. The dataset-dependent training-step test was not run on this compact package.
- The studio GUI constructed and processed an initial UI update. No claim of exhaustive interactive GUI testing is made.
- The packaged generator created one independent synthetic pore scene at 64x, including TOPO, geometric mask, intermediate arrays, metadata and previews. This scene is outside the research data splits.
- Common credential patterns and private-key headers were scanned without matches. No agent instruction files, environment secrets or virtual-environment directories are included.

The full training campaign and historical dataset-dependent test suites were not rerun. Full research image populations and intermediate score caches are not bundled. The validation establishes delivery and example-inference behaviour, not physical image-model validity.
