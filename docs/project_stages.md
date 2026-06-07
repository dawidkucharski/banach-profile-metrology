# Project Stages for Banach-Space Interferometric Repeatability

This document treats the stages as the whole-project workflow, not only as manuscript subsections. The manuscript should report the same logic, but the project also contains executable code, generated outputs and traceable configuration.

## Scopus-Based Research Gap

The Scopus bibliography supports four connected observations. Interferometric metrology has mature tools for stitching, registration and systematic-error compensation in spherical and optical surfaces. Repeatability is still often reported through scalar RMS/PV quantities or pixel-wise standard-deviation matrices. ISO/GUM-oriented uncertainty propagation for areal texture parameters exists, especially through metrological-characteristics thinking, but it is usually applied to classical field parameters rather than function-space distances. Functional data analysis provides a natural language for curves, images and function-valued observations, yet it remains experimentally underused in high-volume industrial interferometric surface metrology.

The project gap is therefore precise: there is no validated workflow that compares classical ISO repeatability and pixel-wise statistics with Banach-space repeatability on both a smooth spherical artefact and a discontinuous step-height artefact using large real interferometric datasets.

## Stage 1: Research Problem

Define the measurand and hypothesis space before analysis. The core question is whether repeatability of interferometric surface maps changes when the observation is treated as a complete function rather than as a collection of scalar descriptors or independent pixels.

## Stage 2: Processing Pipeline

Convert raw interferograms or height maps into aligned functional surfaces. The required operations are import, masking, levelling, detrending, filtering, optional registration and conversion to a common-domain representation `f(x,y)`.

## Stage 3: Banach-Space Framework

Represent each accepted surface or residual as an element of `L1`, `L2`, `Linf`, `W^{1,2}` or `BV`, depending on the metrological risk being tested. The functional repeatability metric is the empirical expectation of the normed distance to the mean function.

## Stage 4: Publication-Ready Code

The R workflow is implemented in `R/btri_repeatability.R` and can process real PNG, TIFF, CSV or RDS surface maps. It computes Ra, Rq, pixel-wise standard-deviation maps, L2 distances to the mean function and functional repeatability. It can simulate data only as a validation fallback; real interferometric folders are the primary pathway. For large frames, polynomial form fits use deterministic spatial subsampling of valid pixels and then evaluate the fitted form over the full image; final residuals and descriptors still use all pixels.

## Stage 5: Repeatability Experiment

Run the same analysis on the spherical ceramic artefact and the step-height steel standard. Compare classical metrics, pixel-wise maps and functional L2 repeatability. Outputs include CSV tables, pixel-wise SD maps, classical-versus-functional scatter plots, boxplots and an interpretation file.

## Stage 6: Interpretation

Classical methods are expected to perform best for certified scalar measurands, homogeneous noise and direct ISO reporting. Functional methods are expected to perform best when repeatability loss is spatially structured, edge-localised, non-Gaussian or morphology-dependent.

## Stage 7: Paper Integration

The paper should present the Scopus-grounded gap in the Introduction, the seven-stage workflow in Methods, real/simulated experiment outputs in Results, a norm-by-norm interpretation in Discussion and a concise conclusion identifying where the functional approach adds metrological value.

## R Execution

Pilot run on real datasets:

```bash
Rscript scripts/run_btri_repeatability.R --ceramic-dir ceramic --steel-dir steel --max-files 20 --selection stratified --output-dir results/r_btri_pilot
```

Full real-data run:

```bash
Rscript scripts/run_btri_repeatability.R --ceramic-dir ceramic --steel-dir steel --max-files 7500 --selection stratified --output-dir results/r_btri_stage5_15000 --no-simulate
```

Completed full real-data run, 15,000 frames total:

- Output folder: `results/r_btri_stage5_15000`
- Ceramic sphere: n = 7500, mean Rq = 0.276591, ISO repeatability limit for Rq = 0.003138, pixel-wise SD RMS = 0.242158, L2 functional repeatability = 0.241967.
- Steel step: n = 7500, mean Rq = 0.216857, ISO repeatability limit for Rq = 0.039103, pixel-wise SD RMS = 0.213740, L2 functional repeatability = 0.213220.
- Frame-level correlations between Rq and L2 distance to the mean residual: ceramic sphere = 0.1519, steel step = 0.9743.
- Values are in native PNG image-intensity residual units until a phase-to-height calibration is supplied.

Simulation fallback, only for software validation when real data are not accessible:

```bash
Rscript scripts/run_btri_repeatability.R --ceramic-dir missing --steel-dir missing --max-files 50 --output-dir results/r_btri_simulation
```