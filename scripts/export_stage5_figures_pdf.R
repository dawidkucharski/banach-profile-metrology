source("R/btri_repeatability.R")

input_dir <- "results/r_btri_stage5_15000"
plot_dir <- file.path(input_dir, "plots")

comparison <- list(
  metrics = utils::read.csv(file.path(input_dir, "combined_frame_metrics.csv")),
  summaries = utils::read.csv(file.path(input_dir, "combined_summary.csv")),
  correlations = utils::read.csv(file.path(input_dir, "classical_functional_correlations.csv"))
)

result_list <- list(
  spherical = list(sd_surface = readRDS(file.path(input_dir, "spherical", "spherical_pixelwise_sd.rds"))),
  step = list(sd_surface = readRDS(file.path(input_dir, "step", "step_pixelwise_sd.rds")))
)

plot_repeatability_outputs(comparison, result_list, plot_dir)