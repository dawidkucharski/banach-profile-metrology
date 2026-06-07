#!/usr/bin/env Rscript

parse_args <- function(args) {
  parsed <- list(
    ceramic_dir = "ceramic",
    steel_dir = "steel",
    output_dir = "results/r_btri_repeatability",
    max_files = 100L,
    selection = "stratified",
    simulate_if_missing = TRUE
  )
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    value <- if (i < length(args)) args[[i + 1L]] else NA_character_
    if (key == "--ceramic-dir") parsed$ceramic_dir <- value
    if (key == "--steel-dir") parsed$steel_dir <- value
    if (key == "--output-dir") parsed$output_dir <- value
    if (key == "--max-files") parsed$max_files <- as.integer(value)
    if (key == "--selection") parsed$selection <- value
    if (key == "--no-simulate") {
      parsed$simulate_if_missing <- FALSE
      i <- i + 1L
      next
    }
    i <- i + 2L
  }
  parsed
}

command_line <- commandArgs(trailingOnly = FALSE)
file_argument <- command_line[grep("^--file=", command_line)]
if (length(file_argument) > 0L) {
  script_path <- normalizePath(sub("^--file=", "", file_argument[[1L]]), mustWork = TRUE)
} else {
  script_path <- normalizePath(file.path("scripts", "run_btri_repeatability.R"), mustWork = FALSE)
}
project_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)
source(file.path(project_root, "R", "btri_repeatability.R"))

arguments <- parse_args(commandArgs(trailingOnly = TRUE))
result <- run_repeatability_experiment(
  ceramic_dir = arguments$ceramic_dir,
  steel_dir = arguments$steel_dir,
  output_dir = arguments$output_dir,
  max_files = arguments$max_files,
  selection = arguments$selection,
  simulate_if_missing = arguments$simulate_if_missing
)

cat("BTRI R repeatability outputs:", normalizePath(arguments$output_dir, mustWork = FALSE), "\n")
print(result$comparison$summaries)
cat("\n", result$interpretation, "\n", sep = "")
