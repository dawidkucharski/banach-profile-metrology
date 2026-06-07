# Banach-space repeatability analysis for interferometric surface maps.
# The functions in this file are designed for real 2D surface maps as well as
# controlled simulations used for method validation.

`%||%` <- function(lhs, rhs) {
  if (is.null(lhs)) rhs else lhs
}

normalise_path <- function(path) {
  normalizePath(path, mustWork = FALSE)
}

numeric_file_index <- function(path) {
  stem <- tools::file_path_sans_ext(basename(path))
  value <- suppressWarnings(as.integer(stem))
  ifelse(is.na(value), NA_integer_, value)
}

discover_surface_files <- function(directory,
                                   pattern = "\\.(png|tif|tiff|csv|rds)$",
                                   selection = c("all", "first", "stratified"),
                                   max_files = NULL,
                                   start_index = NULL,
                                   stop_index = NULL) {
  selection <- match.arg(selection)
  if (!dir.exists(directory)) {
    stop("Surface-map directory does not exist: ", directory, call. = FALSE)
  }
  files <- list.files(directory, pattern = pattern, full.names = TRUE, ignore.case = TRUE)
  if (length(files) == 0L) {
    stop("No supported surface-map files found in: ", directory, call. = FALSE)
  }
  indices <- numeric_file_index(files)
  order_key <- ifelse(is.na(indices), seq_along(files), indices)
  files <- files[order(order_key, files)]
  indices <- numeric_file_index(files)
  keep <- rep(TRUE, length(files))
  if (!is.null(start_index)) keep <- keep & (is.na(indices) | indices >= start_index)
  if (!is.null(stop_index)) keep <- keep & (is.na(indices) | indices <= stop_index)
  files <- files[keep]
  if (length(files) == 0L) {
    stop("No files remain after index filtering.", call. = FALSE)
  }
  if (!is.null(max_files) && max_files < length(files)) {
    if (selection == "first") {
      files <- files[seq_len(max_files)]
    } else if (selection == "stratified") {
      positions <- unique(round(seq(1, length(files), length.out = max_files)))
      files <- files[positions]
    } else {
      files <- files[seq_len(max_files)]
    }
  }
  files
}

read_surface_map <- function(path) {
  extension <- tolower(tools::file_ext(path))
  if (extension == "png") {
    if (!requireNamespace("png", quietly = TRUE)) {
      stop("Package 'png' is required to read PNG interferograms.", call. = FALSE)
    }
    image <- png::readPNG(path)
    if (length(dim(image)) == 3L) {
      image <- apply(image[, , seq_len(min(3L, dim(image)[3L])), drop = FALSE], c(1, 2), mean)
    }
    return(storage.mode_matrix(image))
  }
  if (extension %in% c("tif", "tiff")) {
    if (!requireNamespace("tiff", quietly = TRUE)) {
      stop("Package 'tiff' is required to read TIFF height maps.", call. = FALSE)
    }
    image <- tiff::readTIFF(path)
    if (length(dim(image)) == 3L) {
      image <- apply(image[, , seq_len(min(3L, dim(image)[3L])), drop = FALSE], c(1, 2), mean)
    }
    return(storage.mode_matrix(image))
  }
  if (extension == "csv") {
    return(storage.mode_matrix(as.matrix(utils::read.csv(path, header = FALSE))))
  }
  if (extension == "rds") {
    object <- readRDS(path)
    return(storage.mode_matrix(as.matrix(object)))
  }
  stop("Unsupported surface-map extension: ", extension, call. = FALSE)
}

storage.mode_matrix <- function(x) {
  matrix_x <- as.matrix(x)
  storage.mode(matrix_x) <- "double"
  matrix_x
}

valid_surface_mask <- function(surface) {
  is.finite(surface)
}

coordinate_grid <- function(surface) {
  nr <- nrow(surface)
  nc <- ncol(surface)
  list(
    x = matrix(rep(seq(-1, 1, length.out = nc), each = nr), nrow = nr, ncol = nc),
    y = matrix(rep(seq(-1, 1, length.out = nr), times = nc), nrow = nr, ncol = nc)
  )
}

.polynomial_design_cache <- new.env(parent = emptyenv())

polynomial_design <- function(nr, nc, degree, max_fit_pixels = 20000L) {
  key <- paste(nr, nc, degree, max_fit_pixels, sep = "x")
  if (exists(key, envir = .polynomial_design_cache, inherits = FALSE)) {
    return(get(key, envir = .polynomial_design_cache, inherits = FALSE))
  }
  x <- matrix(rep(seq(-1, 1, length.out = nc), each = nr), nrow = nr, ncol = nc)
  y <- matrix(rep(seq(-1, 1, length.out = nr), times = nc), nrow = nr, ncol = nc)
  terms <- list(intercept = rep(1, nr * nc))
  if (degree >= 1L) {
    terms$x <- as.vector(x)
    terms$y <- as.vector(y)
  }
  if (degree >= 2L) {
    terms$x2 <- as.vector(x^2)
    terms$xy <- as.vector(x * y)
    terms$y2 <- as.vector(y^2)
  }
  full_design <- do.call(cbind, terms)
  full_mask <- matrix(TRUE, nr, nc)
  fit_positions <- which(sample_fit_mask(full_mask, max_fit_pixels = max_fit_pixels))
  design <- list(full = full_design, fit_positions = fit_positions)
  assign(key, design, envir = .polynomial_design_cache)
  design
}

sample_fit_mask <- function(mask, max_fit_pixels = 20000L) {
  finite_count <- sum(mask, na.rm = TRUE)
  if (finite_count <= max_fit_pixels) {
    return(mask)
  }
  stride <- ceiling(sqrt(finite_count / max_fit_pixels))
  sampled <- matrix(FALSE, nrow(mask), ncol(mask))
  sampled[seq(1L, nrow(mask), by = stride), seq(1L, ncol(mask), by = stride)] <- TRUE
  fit_mask <- mask & sampled
  if (sum(fit_mask, na.rm = TRUE) < min(max_fit_pixels / 4L, finite_count)) {
    positions <- which(mask)
    selected <- positions[unique(round(seq(1L, length(positions), length.out = min(max_fit_pixels, length(positions)))))]
    fit_mask <- matrix(FALSE, nrow(mask), ncol(mask))
    fit_mask[selected] <- TRUE
  }
  fit_mask
}

fit_polynomial_surface <- function(surface, degree = 1L, mask = NULL, max_fit_pixels = 20000L) {
  if (!(degree %in% c(0L, 1L, 2L))) {
    stop("Only polynomial degrees 0, 1 and 2 are supported.", call. = FALSE)
  }
  mask <- mask %||% valid_surface_mask(surface)
  cached <- polynomial_design(nrow(surface), ncol(surface), degree, max_fit_pixels = max_fit_pixels)
  fit_positions <- cached$fit_positions[mask[cached$fit_positions]]
  finite_count <- sum(mask, na.rm = TRUE)
  if (length(fit_positions) < min(max_fit_pixels / 4L, finite_count)) {
    fit_positions <- which(sample_fit_mask(mask, max_fit_pixels = max_fit_pixels))
  }
  surface_vector <- as.vector(surface)
  design <- cached$full[fit_positions, , drop = FALSE]
  response <- surface_vector[fit_positions]
  fit <- stats::lm.fit(design, response)
  full_design <- cached$full
  fitted_surface <- matrix(as.vector(full_design %*% fit$coefficients), nrow(surface), ncol(surface))
  list(model = fitted_surface, residual = surface - fitted_surface, coefficients = fit$coefficients)
}

level_surface <- function(surface, mask = NULL) {
  fit_polynomial_surface(surface, degree = 1L, mask = mask)
}

detect_step_edge <- function(surface, axis = c("x", "y"), min_margin = 12L) {
  axis <- match.arg(axis)
  profile <- if (axis == "x") colMeans(surface, na.rm = TRUE) else rowMeans(surface, na.rm = TRUE)
  if (length(profile) <= 2L * min_margin + 2L) {
    return(max(2L, round(length(profile) / 2)))
  }
  gradient <- diff(profile)
  interior <- seq.int(min_margin, length(gradient) - min_margin)
  if (length(interior) == 0L || all(!is.finite(gradient[interior]))) {
    return(round(length(profile) / 2))
  }
  interior[which.max(abs(gradient[interior]))] + 1L
}

step_masks <- function(surface, edge_index, axis = c("x", "y"), edge_band = 8L) {
  axis <- match.arg(axis)
  nr <- nrow(surface)
  nc <- ncol(surface)
  if (axis == "x") {
    coordinate <- matrix(rep(seq_len(nc), each = nr), nrow = nr, ncol = nc)
  } else {
    coordinate <- matrix(rep(seq_len(nr), times = nc), nrow = nr, ncol = nc)
  }
  left <- coordinate < (edge_index - edge_band)
  right <- coordinate > (edge_index + edge_band)
  edge <- !(left | right)
  list(left = left, right = right, edge = edge)
}

detrend_surface <- function(surface,
                            artefact = c("spherical", "step"),
                            mask = NULL,
                            step_axis = "x",
                            edge_band = 8L) {
  artefact <- match.arg(artefact)
  mask <- mask %||% valid_surface_mask(surface)
  if (artefact == "spherical") {
    fit <- fit_polynomial_surface(surface, degree = 2L, mask = mask)
    return(list(
      model = fit$model,
      residual = fit$residual,
      parameters = as.list(fit$coefficients),
      domains = list(valid = mask)
    ))
  }
  edge_index <- detect_step_edge(surface, axis = step_axis, min_margin = edge_band + 2L)
  edge_index <- max(edge_band + 2L, min(edge_index, if (step_axis == "x") ncol(surface) - edge_band - 1L else nrow(surface) - edge_band - 1L))
  domains <- step_masks(surface, edge_index = edge_index, axis = step_axis, edge_band = edge_band)
  finite <- valid_surface_mask(surface) & mask
  left_mask <- domains$left & finite
  right_mask <- domains$right & finite
  if (!any(left_mask) || !any(right_mask)) {
    edge_index <- if (step_axis == "x") round(ncol(surface) / 2) else round(nrow(surface) / 2)
    domains <- step_masks(surface, edge_index = edge_index, axis = step_axis, edge_band = edge_band)
    left_mask <- domains$left & finite
    right_mask <- domains$right & finite
  }
  if (!any(left_mask) || !any(right_mask)) {
    stop("Step model has an empty plateau region even after central fallback; adjust edge_band or mask.", call. = FALSE)
  }
  left_level <- stats::median(surface[left_mask], na.rm = TRUE)
  right_level <- stats::median(surface[right_mask], na.rm = TRUE)
  model <- matrix(0, nrow(surface), ncol(surface))
  model[domains$left] <- left_level
  model[domains$right] <- right_level
  model[domains$edge] <- 0.5 * (left_level + right_level)
  list(
    model = model,
    residual = surface - model,
    parameters = list(edge_index = edge_index, left_level = left_level, right_level = right_level, step_height = right_level - left_level),
    domains = domains
  )
}

gaussian_kernel <- function(sigma, radius = ceiling(3 * sigma)) {
  x <- seq(-radius, radius)
  kernel <- exp(-(x^2) / (2 * sigma^2))
  kernel / sum(kernel)
}

gaussian_smooth <- function(surface, sigma = 1) {
  if (sigma <= 0) return(surface)
  kernel <- gaussian_kernel(sigma)
  row_filtered <- t(apply(surface, 1, stats::filter, filter = kernel, sides = 2, circular = FALSE))
  col_filtered <- apply(row_filtered, 2, stats::filter, filter = kernel, sides = 2, circular = FALSE)
  smoothed <- matrix(as.numeric(col_filtered), nrow(surface), ncol(surface))
  smoothed[is.na(smoothed)] <- surface[is.na(smoothed)]
  smoothed
}

preprocess_surface <- function(surface,
                               artefact = c("spherical", "step"),
                               do_level = TRUE,
                               filter_sigma = NULL,
                               step_axis = "x",
                               edge_band = 8L) {
  artefact <- match.arg(artefact)
  mask <- valid_surface_mask(surface)
  working <- surface
  level <- NULL
  if (do_level) {
    level <- level_surface(working, mask = mask)
    working <- level$residual
  }
  if (!is.null(filter_sigma) && filter_sigma > 0) {
    working <- working - gaussian_smooth(working, sigma = filter_sigma)
  }
  detrended <- detrend_surface(working, artefact = artefact, mask = mask, step_axis = step_axis, edge_band = edge_band)
  list(surface = working, residual = detrended$residual, model = detrended$model, mask = mask, level = level, detrend = detrended)
}

roughness_metrics <- function(surface, mask = NULL) {
  mask <- mask %||% valid_surface_mask(surface)
  values <- surface[mask]
  values <- values[is.finite(values)]
  centred <- values - mean(values)
  c(
    Ra = mean(abs(centred)),
    Rq = sqrt(mean(centred^2)),
    min = min(values),
    max = max(values),
    pv = max(values) - min(values)
  )
}

l2_norm_difference <- function(surface_a, surface_b, mask = NULL, pixel_area = 1) {
  if (!all(dim(surface_a) == dim(surface_b))) {
    stop("Surface dimensions must match for L2 distance.", call. = FALSE)
  }
  mask <- mask %||% (valid_surface_mask(surface_a) & valid_surface_mask(surface_b))
  difference <- surface_a[mask] - surface_b[mask]
  sqrt(mean(difference^2) * pixel_area)
}

pixelwise_welford_update <- function(state, surface) {
  if (is.null(state)) {
    return(list(n = 1L, mean = surface, m2 = matrix(0, nrow(surface), ncol(surface))))
  }
  if (!all(dim(state$mean) == dim(surface))) {
    stop("All surfaces in a repeatability run must have identical dimensions after registration.", call. = FALSE)
  }
  n_new <- state$n + 1L
  delta <- surface - state$mean
  mean_new <- state$mean + delta / n_new
  delta2 <- surface - mean_new
  list(n = n_new, mean = mean_new, m2 = state$m2 + delta * delta2)
}

pixelwise_sd_from_state <- function(state) {
  if (is.null(state) || state$n < 2L) {
    return(matrix(NA_real_, nrow(state$mean), ncol(state$mean)))
  }
  sqrt(state$m2 / (state$n - 1L))
}

simulate_surface_maps <- function(artefact = c("spherical", "step"),
                                  n = 50L,
                                  dims = c(96L, 128L),
                                  noise_sd = 0.01,
                                  drift_sd = 0.002,
                                  step_height = 0.4,
                                  seed = 1L) {
  artefact <- match.arg(artefact)
  set.seed(seed)
  nr <- dims[1L]
  nc <- dims[2L]
  x <- matrix(rep(seq(-1, 1, length.out = nc), each = nr), nrow = nr, ncol = nc)
  y <- matrix(rep(seq(-1, 1, length.out = nr), times = nc), nrow = nr, ncol = nc)
  base <- if (artefact == "spherical") {
    0.2 * (x^2 + y^2)
  } else {
    ifelse(x > 0, step_height, 0)
  }
  maps <- vector("list", n)
  for (i in seq_len(n)) {
    drift <- stats::rnorm(1L, sd = drift_sd)
    noise <- matrix(stats::rnorm(nr * nc, sd = noise_sd), nr, nc)
    if (artefact == "step") {
      edge_jitter <- round(stats::rnorm(1L, sd = 1.5))
      shifted_x <- matrix(rep(seq_len(nc) - nc / 2 - edge_jitter, each = nr), nr, nc)
      base_i <- ifelse(shifted_x > 0, step_height, 0)
    } else {
      base_i <- base
    }
    maps[[i]] <- base_i + drift + noise
  }
  maps
}

process_surface_files <- function(files,
                                  artefact = c("spherical", "step"),
                                  output_dir,
                                  filter_sigma = NULL,
                                  step_axis = "x",
                                  edge_band = 8L,
                                  pixel_area = 1,
                                  progress_interval = 500L) {
  artefact <- match.arg(artefact)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  state <- NULL
  classical_rows <- vector("list", length(files))
  for (i in seq_along(files)) {
    if (i == 1L || i == length(files) || i %% progress_interval == 0L) {
      message(sprintf("%s first pass: %d/%d", artefact, i, length(files)))
    }
    raw <- read_surface_map(files[[i]])
    processed <- preprocess_surface(raw, artefact = artefact, filter_sigma = filter_sigma, step_axis = step_axis, edge_band = edge_band)
    residual <- processed$residual
    state <- pixelwise_welford_update(state, residual)
    metrics <- roughness_metrics(residual, mask = processed$mask)
    extra <- if (artefact == "step") processed$detrend$parameters$step_height else NA_real_
    classical_rows[[i]] <- data.frame(
      artefact = artefact,
      file = basename(files[[i]]),
      sequence_index = numeric_file_index(files[[i]]),
      Ra = metrics[["Ra"]],
      Rq = metrics[["Rq"]],
      PV = metrics[["pv"]],
      step_height = extra,
      stringsAsFactors = FALSE
    )
  }
  mean_surface <- state$mean
  sd_surface <- pixelwise_sd_from_state(state)
  functional_rows <- vector("list", length(files))
  for (i in seq_along(files)) {
    if (i == 1L || i == length(files) || i %% progress_interval == 0L) {
      message(sprintf("%s second pass: %d/%d", artefact, i, length(files)))
    }
    raw <- read_surface_map(files[[i]])
    processed <- preprocess_surface(raw, artefact = artefact, filter_sigma = filter_sigma, step_axis = step_axis, edge_band = edge_band)
    residual <- processed$residual
    l2_to_mean <- l2_norm_difference(residual, mean_surface, mask = processed$mask, pixel_area = pixel_area)
    functional_rows[[i]] <- data.frame(
      artefact = artefact,
      file = basename(files[[i]]),
      sequence_index = numeric_file_index(files[[i]]),
      L2_to_mean = l2_to_mean,
      stringsAsFactors = FALSE
    )
  }
  classical <- do.call(rbind, classical_rows)
  functional <- do.call(rbind, functional_rows)
  frame_metrics <- merge(classical, functional, by = c("artefact", "file", "sequence_index"), all = TRUE)
  functional_repeatability <- mean(frame_metrics$L2_to_mean, na.rm = TRUE)
  functional_repeatability_rms <- sqrt(mean(frame_metrics$L2_to_mean^2, na.rm = TRUE))
  pixelwise_sd_rms <- sqrt(mean(sd_surface^2, na.rm = TRUE))
  summary <- data.frame(
    artefact = artefact,
    n = length(files),
    mean_Ra = mean(frame_metrics$Ra, na.rm = TRUE),
    sd_Ra = stats::sd(frame_metrics$Ra, na.rm = TRUE),
    mean_Rq = mean(frame_metrics$Rq, na.rm = TRUE),
    sd_Rq = stats::sd(frame_metrics$Rq, na.rm = TRUE),
    iso_repeatability_limit_Rq = 2.8 * stats::sd(frame_metrics$Rq, na.rm = TRUE),
    pixelwise_sd_rms = pixelwise_sd_rms,
    functional_repeatability_L2 = functional_repeatability,
    functional_repeatability_L2_rms = functional_repeatability_rms,
    stringsAsFactors = FALSE
  )
  utils::write.csv(frame_metrics, file.path(output_dir, paste0(artefact, "_frame_metrics.csv")), row.names = FALSE)
  utils::write.csv(summary, file.path(output_dir, paste0(artefact, "_summary.csv")), row.names = FALSE)
  saveRDS(mean_surface, file.path(output_dir, paste0(artefact, "_mean_surface.rds")))
  saveRDS(sd_surface, file.path(output_dir, paste0(artefact, "_pixelwise_sd.rds")))
  list(metrics = frame_metrics, summary = summary, mean_surface = mean_surface, sd_surface = sd_surface)
}

process_surface_list <- function(surface_list,
                                 artefact = c("spherical", "step"),
                                 output_dir,
                                 filter_sigma = NULL,
                                 step_axis = "x",
                                 edge_band = 8L,
                                 pixel_area = 1) {
  artefact <- match.arg(artefact)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  state <- NULL
  residuals <- vector("list", length(surface_list))
  classical_rows <- vector("list", length(surface_list))
  for (i in seq_along(surface_list)) {
    processed <- preprocess_surface(surface_list[[i]], artefact = artefact, filter_sigma = filter_sigma, step_axis = step_axis, edge_band = edge_band)
    residuals[[i]] <- processed$residual
    state <- pixelwise_welford_update(state, processed$residual)
    metrics <- roughness_metrics(processed$residual, mask = processed$mask)
    extra <- if (artefact == "step") processed$detrend$parameters$step_height else NA_real_
    classical_rows[[i]] <- data.frame(
      artefact = artefact,
      file = paste0("sim_", sprintf("%05d", i), ".rds"),
      sequence_index = i,
      Ra = metrics[["Ra"]],
      Rq = metrics[["Rq"]],
      PV = metrics[["pv"]],
      step_height = extra,
      stringsAsFactors = FALSE
    )
  }
  mean_surface <- state$mean
  sd_surface <- pixelwise_sd_from_state(state)
  functional_rows <- lapply(seq_along(residuals), function(i) {
    data.frame(
      artefact = artefact,
      file = paste0("sim_", sprintf("%05d", i), ".rds"),
      sequence_index = i,
      L2_to_mean = l2_norm_difference(residuals[[i]], mean_surface, pixel_area = pixel_area),
      stringsAsFactors = FALSE
    )
  })
  frame_metrics <- merge(do.call(rbind, classical_rows), do.call(rbind, functional_rows), by = c("artefact", "file", "sequence_index"), all = TRUE)
  summary <- data.frame(
    artefact = artefact,
    n = length(surface_list),
    mean_Ra = mean(frame_metrics$Ra, na.rm = TRUE),
    sd_Ra = stats::sd(frame_metrics$Ra, na.rm = TRUE),
    mean_Rq = mean(frame_metrics$Rq, na.rm = TRUE),
    sd_Rq = stats::sd(frame_metrics$Rq, na.rm = TRUE),
    iso_repeatability_limit_Rq = 2.8 * stats::sd(frame_metrics$Rq, na.rm = TRUE),
    pixelwise_sd_rms = sqrt(mean(sd_surface^2, na.rm = TRUE)),
    functional_repeatability_L2 = mean(frame_metrics$L2_to_mean, na.rm = TRUE),
    functional_repeatability_L2_rms = sqrt(mean(frame_metrics$L2_to_mean^2, na.rm = TRUE)),
    stringsAsFactors = FALSE
  )
  utils::write.csv(frame_metrics, file.path(output_dir, paste0(artefact, "_frame_metrics.csv")), row.names = FALSE)
  utils::write.csv(summary, file.path(output_dir, paste0(artefact, "_summary.csv")), row.names = FALSE)
  saveRDS(mean_surface, file.path(output_dir, paste0(artefact, "_mean_surface.rds")))
  saveRDS(sd_surface, file.path(output_dir, paste0(artefact, "_pixelwise_sd.rds")))
  list(metrics = frame_metrics, summary = summary, mean_surface = mean_surface, sd_surface = sd_surface)
}

compare_repeatability <- function(result_list, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  metrics <- do.call(rbind, lapply(result_list, `[[`, "metrics"))
  summaries <- do.call(rbind, lapply(result_list, `[[`, "summary"))
  safe_cor <- function(x, y) {
    complete <- is.finite(x) & is.finite(y)
    if (sum(complete) < 3L) return(NA_real_)
    if (stats::sd(x[complete]) == 0 || stats::sd(y[complete]) == 0) return(NA_real_)
    stats::cor(x[complete], y[complete])
  }
  correlation <- by(metrics, metrics$artefact, function(group) {
    data.frame(
      artefact = group$artefact[1L],
      cor_Rq_L2 = safe_cor(group$Rq, group$L2_to_mean),
      cor_Ra_L2 = safe_cor(group$Ra, group$L2_to_mean),
      stringsAsFactors = FALSE
    )
  })
  correlation <- do.call(rbind, correlation)
  utils::write.csv(metrics, file.path(output_dir, "combined_frame_metrics.csv"), row.names = FALSE)
  utils::write.csv(summaries, file.path(output_dir, "combined_summary.csv"), row.names = FALSE)
  utils::write.csv(correlation, file.path(output_dir, "classical_functional_correlations.csv"), row.names = FALSE)
  list(metrics = metrics, summaries = summaries, correlations = correlation)
}

open_plot_device <- function(path, width = 8, height = 6, res = 150) {
  extension <- tolower(tools::file_ext(path))
  if (extension == "pdf") {
    grDevices::pdf(path, width = width, height = height, useDingbats = FALSE)
  } else {
    grDevices::png(path, width = width * res, height = height * res, res = res)
  }
}

plot_matrix <- function(matrix_data, path, title) {
  open_plot_device(path, width = 8, height = 6)
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit({
    graphics::par(old_par)
    grDevices::dev.off()
  }, add = TRUE)
  graphics::image(t(matrix_data[nrow(matrix_data):1, ]), col = grDevices::hcl.colors(256, "Viridis"), axes = FALSE, main = title)
}

plot_repeatability_outputs <- function(comparison, result_list, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  metrics <- comparison$metrics
  for (extension in c("pdf", "png")) {
    open_plot_device(file.path(output_dir, paste0("classical_vs_functional_scatter.", extension)), width = 8, height = 6)
    graphics::plot(metrics$Rq, metrics$L2_to_mean, col = as.factor(metrics$artefact), pch = 19,
                   xlab = "Classical Rq", ylab = "Functional L2 distance to mean",
                   main = "Classical versus functional repeatability")
    graphics::legend("topleft", legend = levels(as.factor(metrics$artefact)), col = seq_along(levels(as.factor(metrics$artefact))), pch = 19)
    grDevices::dev.off()
    open_plot_device(file.path(output_dir, paste0("functional_repeatability_boxplot.", extension)), width = 8, height = 6)
    graphics::boxplot(L2_to_mean ~ artefact, data = metrics, ylab = "L2 distance to mean", main = "Functional repeatability by geometry")
    grDevices::dev.off()
  }
  for (name in names(result_list)) {
    for (extension in c("pdf", "png")) {
      plot_matrix(result_list[[name]]$sd_surface, file.path(output_dir, paste0(name, "_pixelwise_sd_map.", extension)), paste(name, "pixel-wise standard deviation"))
    }
  }
}

interpret_repeatability <- function(comparison) {
  summaries <- comparison$summaries
  smooth <- summaries[summaries$artefact == "spherical", , drop = FALSE]
  step <- summaries[summaries$artefact == "step", , drop = FALSE]
  lines <- c(
    "Interpretation of classical and functional repeatability comparison",
    "",
    "Classical metrics outperform when the target measurand is a certified scalar quantity, when direct ISO repeatability reporting is required, or when the dominant variation is spatially homogeneous Gaussian noise. In that case Rq and the pixel-wise standard-deviation map are transparent and easy to audit.",
    "",
    "Functional metrics outperform when repeatability loss has geometry, for example edge motion in the step-height standard, localised defects, non-Gaussian excursions, or spatially structured form-removal residuals. The L2 distance to the mean surface treats each interferogram as one function-valued observation rather than as unrelated pixels.",
    "",
    "On smooth spherical surfaces, Rq and L2 repeatability are expected to be close because both respond to distributed residual energy. On discontinuous step-height surfaces, the functional metric is expected to diverge from scalar roughness metrics because edge-band displacement and plateau imbalance can preserve similar Rq values while changing the shape of the residual function.",
    "",
    sprintf("Spherical functional repeatability L2: %s", if (nrow(smooth)) signif(smooth$functional_repeatability_L2, 6) else "not estimated"),
    sprintf("Step functional repeatability L2: %s", if (nrow(step)) signif(step$functional_repeatability_L2, 6) else "not estimated")
  )
  paste(lines, collapse = "\n")
}

run_repeatability_experiment <- function(ceramic_dir = NULL,
                                         steel_dir = NULL,
                                         output_dir = "results/r_btri_repeatability",
                                         max_files = 100L,
                                         selection = "stratified",
                                         simulate_if_missing = TRUE) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  results <- list()
  if (!is.null(ceramic_dir) && dir.exists(ceramic_dir)) {
    ceramic_files <- discover_surface_files(ceramic_dir, selection = selection, max_files = max_files)
    results$spherical <- process_surface_files(ceramic_files, artefact = "spherical", output_dir = file.path(output_dir, "spherical"))
  } else if (simulate_if_missing) {
    maps <- simulate_surface_maps("spherical", n = max_files, seed = 10L)
    results$spherical <- process_surface_list(maps, artefact = "spherical", output_dir = file.path(output_dir, "spherical"))
  }
  if (!is.null(steel_dir) && dir.exists(steel_dir)) {
    steel_files <- discover_surface_files(steel_dir, selection = selection, max_files = max_files)
    results$step <- process_surface_files(steel_files, artefact = "step", output_dir = file.path(output_dir, "step"))
  } else if (simulate_if_missing) {
    maps <- simulate_surface_maps("step", n = max_files, seed = 20L)
    results$step <- process_surface_list(maps, artefact = "step", output_dir = file.path(output_dir, "step"))
  }
  comparison <- compare_repeatability(results, output_dir = output_dir)
  plot_repeatability_outputs(comparison, results, output_dir = file.path(output_dir, "plots"))
  interpretation <- interpret_repeatability(comparison)
  writeLines(interpretation, file.path(output_dir, "interpretation.txt"))
  list(results = results, comparison = comparison, interpretation = interpretation)
}
