# Compute the ATE of eligibility from held-out outcome and propensity predictions.
# Rscript score_aipw.R analysis.csv predictions.csv.gz results_directory
read_data <- function(path) {
  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
}

check_analysis <- function(data) {
  stopifnot(all(c("row_id", "fold", "net_tfa", "e401") %in% names(data)),
            nrow(data) == 9915L, !anyNA(data$row_id), !anyDuplicated(data$row_id),
            all(is.finite(data$net_tfa)), all(data$e401 %in% 0:1),
            sum(data$e401) == 3682L, setequal(data$fold, 1:5),
            all(table(data$fold, data$e401) > 0L))
}

score_one <- function(data, pred, bounds = c(0.01, 0.025, 0.05)) {
  stopifnot(nrow(pred) == nrow(data), !anyDuplicated(pred$row_id),
            setequal(pred$row_id, data$row_id))
  pred <- pred[match(data$row_id, pred$row_id), ]
  stopifnot(all(pred$fold == data$fold), all(is.finite(pred$m0_hat)),
            all(is.finite(pred$m1_hat)), all(pred$e_hat >= 0 & pred$e_hat <= 1))
  y <- data$net_tfa
  treated <- data$e401 == 1L
  e <- pred$e_hat
  mu0 <- pred$m0_hat
  mu1 <- pred$m1_hat
  n <- length(y)
  factual_mean <- ifelse(treated, mu1, mu0)
  diagnostics <- data.frame(
    n = n, n_treated = sum(treated),
    outcome_rmse = sqrt(mean((y - factual_mean)^2)),
    outcome_rmse_0 = sqrt(mean((y[!treated] - mu0[!treated])^2)),
    outcome_rmse_1 = sqrt(mean((y[treated] - mu1[treated])^2)),
    propensity_brier = mean((as.numeric(treated) - e)^2),
    propensity_min = min(e), propensity_p01 = unname(quantile(e, 0.01)),
    propensity_p99 = unname(quantile(e, 0.99)), propensity_max = max(e)
  )
  estimates <- lapply(bounds, function(a) {
    trimmed <- pmin(pmax(e, a), 1 - a)
    # DML2 with observation-weighted pooling across all five assessment folds.
    phi <- mu1 - mu0
    phi[treated] <- phi[treated] + (y[treated] - mu1[treated]) / trimmed[treated]
    phi[!treated] <- phi[!treated] - (y[!treated] - mu0[!treated]) / (1 - trimmed[!treated])
    ate <- mean(phi)
    se <- sqrt(mean((phi - ate)^2) / n)
    z <- qnorm(0.975)
    data.frame(bound = a, ate = ate, se = se, ci_low = ate - z * se,
               ci_high = ate + z * se, n_clip_low = sum(e < a),
               n_clip_high = sum(e > 1 - a))
  })
  list(estimates = do.call(rbind, estimates), diagnostics = diagnostics)
}

score_grid <- function(data, predictions) {
  check_analysis(data)
  keys <- c("depth", "width", "method")
  required <- c(keys, "row_id", "fold", "m0_hat", "m1_hat", "e_hat")
  stopifnot(all(required %in% names(predictions)), !anyNA(predictions[required]),
            all(nzchar(predictions$method)), all(predictions$depth %in% c(2L, 5L, 10L)),
            all(predictions$width %in% c(10L, 25L)), length(unique(predictions$method)) == 6L)
  groups <- split(seq_len(nrow(predictions)), interaction(predictions[keys], drop = TRUE))
  stopifnot(length(groups) == 36L)
  estimates <- list()
  diagnostics <- list()
  for (g in seq_along(groups)) {
    p <- predictions[groups[[g]], ]
    out <- score_one(data, p)
    id <- p[1L, keys, drop = FALSE]
    estimates[[g]] <- cbind(id[rep(1L, nrow(out$estimates)), , drop = FALSE], out$estimates)
    diagnostics[[g]] <- cbind(id, out$diagnostics)
  }
  list(estimates = do.call(rbind, estimates), diagnostics = do.call(rbind, diagnostics))
}

write_results <- function(out, directory) {
  dir.create(directory, showWarnings = FALSE, recursive = TRUE)
  rownames(out$estimates) <- NULL
  rownames(out$diagnostics) <- NULL
  write.csv(out$estimates, file.path(directory, "aipw_estimates.csv"), row.names = FALSE)
  write.csv(out$diagnostics, file.path(directory, "nuisance_diagnostics.csv"), row.names = FALSE)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  stopifnot(length(args) == 3L)
  dat <- read_data(args[1L])
  pred <- read_data(args[2L])
  write_results(score_grid(dat, pred), args[3L])
}
