# Run from the repository root with R and jsonlite:
# Rscript tools/generate-trees-cycle.R          # regenerate the JSON
# Rscript tools/generate-trees-cycle.R --check # compare with the committed values
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 0L || identical(args, "--check"))
if (!file.exists("R/gltbnn.R")) stop("Run this script from the gltbnn source root.")
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Install jsonlite to write or check the example.")

# Use this checkout, including its public fitting function and shrinkage summary.
pkg <- new.env(parent = globalenv())
for (file in list.files("R", pattern = "[.]R$", full.names = TRUE)) sys.source(file, pkg)
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
held_out <- seq(5L, nrow(datasets::trees), by = 5L)
x <- as.matrix(datasets::trees[-held_out, c("Girth", "Height")])
y <- datasets::trees$Volume[-held_out]
hidden <- c(3L, 2L)
seed <- 31L
ctrl <- pkg$.gltbnn_control(list(outer = 1L))
priors <- c("G", "GL", "GLT")
coefficient <- list(layer = 1L, unit = 1L, coordinate = 1L, predictor = "Girth")

same <- function(a, b) {
  stopifnot(isTRUE(all.equal(unname(a), unname(b), tolerance = 1e-10)))
}

snapshot <- function(parameters, phi, local_precision) {
  l <- coefficient$layer
  k <- coefficient$unit
  j <- coefficient$coordinate
  m <- parameters$mu[[l]][k, j]
  s <- exp(parameters$logsd[[l]][k, j])
  list(mean = m, sd = s, second_moment = m^2 + s^2, noise_precision = phi,
       scale_precision = local_precision, effective_precision = phi * local_precision)
}

one_cycle <- function(prior) {
  fit <- pkg$gltbnn_fit(x, y, hidden = hidden, prior = prior,
                        seed = seed, control = list(outer = 1L))
  engine <- fit$engine
  count <- engine$config$M
  x_scaled <- pkg$.gltbnn_scale_x(x, fit$preprocess)
  y_scaled <- (y - fit$preprocess$y_center) / fit$preprocess$y_scale

  # Reconstruct cycle zero using the package initializer, then verify its monitored
  # ELBO against the value saved by the actual fit. No update routine is replaced.
  weights <- pkg$.bnn_initialize(x_scaled, y_scaled, hidden, seed, ctrl$init_steps)
  start <- list(mu = weights, logsd = lapply(weights, function(w) {
    matrix(log(.01), nrow = nrow(w), ncol = ncol(w))
  }))
  groups <- lapply(count, function(n) pkg$.bnn_init_groups(n, 1L, prior, "coefficient"))
  old <- list(g = groups, noise = pkg$.bnn_ig(3, 1), u = pkg$.bnn_ig(2, 2),
              rho = pkg$.bnn_ig(1, 1.04))
  old_terms <- Map(function(g, n) pkg$.bnn_prior_terms(g, n, prior), groups, count)
  set.seed(seed + 1L)
  monitor <- pkg$.bnn_antithetic_draws(start, ctrl$monitor_draws)
  elbo0 <- pkg$.bnn_elbo(old, start, x_scaled, y_scaled, old_terms, monitor)
  same(elbo0$value, engine$trace$elbo[1L])

  # The optimizer draws one new antithetic bank per weight proposal; backtracking
  # reuses that bank. Replay their RNG consumption to recover the actual 16 residual
  # draws. Only the matrix dimensions matter for drawing Gaussian perturbations.
  for (j in seq_len(ctrl$weight_steps)) {
    invisible(pkg$.bnn_antithetic_draws(start, ctrl$weight_draws))
  }
  end <- engine$parameters
  residual_draws <- pkg$.bnn_antithetic_draws(end, ctrl$noise_draws)
  expected_sse <- mean(pkg$.bnn_sse(end, x_scaled, y_scaled, residual_draws))
  moments <- pkg$.bnn_weight_moments(end, coefficient = TRUE)
  hidden_penalty <- sum(vapply(seq_along(count), function(l) {
    sum(moments[[l]] * old_terms[[l]]$precision)
  }, numeric(1)))
  noise <- engine$state$noise
  same(noise$a, (nrow(x) + sum(count)) / 2)
  same(noise$b, (expected_sse + hidden_penalty) / 2)

  l <- coefficient$layer
  index <- coefficient$unit + (coefficient$coordinate - 1L) * hidden[l]
  scales <- pkg$gltbnn_shrinkage(fit)
  selected <- with(scales, layer == coefficient$layer & unit == coefficient$unit &
                     coordinate == coefficient$coordinate)
  stopifnot(sum(selected) == 1L)
  before <- snapshot(start, old$noise$phi, old_terms[[l]]$precision[index])
  after <- snapshot(end, noise$phi, scales$scale_precision[selected])
  same(after$second_moment, moments[[l]][index])
  g <- engine$state$g[[l]]
  scale_update <- switch(prior,
    G = list(layer_inverse_squared_scale = g$tau$phi),
    GL = list(layer_inverse_squared_scale = g$tau$phi,
              multiplier_inverse_squared_scale = g$local$phi[index]),
    GLT = list(kernel_A = groups[[l]]$tau$phi * groups[[l]]$nu$phi[index],
               kernel_B = noise$phi * moments[[l]][index],
               mean_local_scale = g$local$u[index],
               geometric_tail_shape = exp(g$tail$eta_mean))
  )
  reproduced_scale <- switch(prior,
    G = g$tau$phi,
    GL = g$tau$phi * g$local$phi[index],
    GLT = pkg$.bnn_scalar_locals(scale_update$kernel_A, scale_update$kernel_B)$phi
  )
  same(after$scale_precision, reproduced_scale)
  list(before = before, after = after,
       update = list(expected_sse = expected_sse, hidden_penalty = hidden_penalty,
                     noise_shape = noise$a, noise_scale = noise$b,
                     accepted_steps = sum(engine$weight_steps$accepted),
                     layer_second_moment_sum = sum(moments[[l]]),
                     scale_update = scale_update))
}

fixture <- list(
  dataset = "datasets::trees", n_train = nrow(x), held_out_rows = held_out,
  predictors = colnames(x), response = "Volume", coefficient = coefficient,
  control = list(hidden = hidden, seed = seed, init_steps = ctrl$init_steps,
                 outer = ctrl$outer, weight_steps = ctrl$weight_steps,
                 learning_rate = ctrl$lr, weight_draws = ctrl$weight_draws,
                 noise_draws = ctrl$noise_draws, monitor_draws = ctrl$monitor_draws,
                 endpoint_draws = ctrl$endpoint_draws),
  priors = setNames(lapply(priors, one_cycle), priors)
)
target <- "docs/data/trees-cycle.json"
json <- jsonlite::toJSON(fixture, auto_unbox = TRUE, digits = 12, pretty = TRUE)
if (identical(args, "--check")) {
  reference <- jsonlite::fromJSON(target, simplifyVector = FALSE)
  generated <- jsonlite::fromJSON(json, simplifyVector = FALSE)
  stopifnot(isTRUE(all.equal(reference, generated, tolerance = 1e-9)))
  message("All three recorded cycles agree with new fits.")
} else {
  writeLines(json, target, useBytes = TRUE)
}
