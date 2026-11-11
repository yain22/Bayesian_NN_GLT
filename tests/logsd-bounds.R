library(gltbnn)

x <- matrix(seq(-1, 1, length.out = 12), ncol = 1L)
y <- as.vector(x^2 + 0.1 * x)
control <- list(init_steps = 2L, outer = 1L, weight_steps = 2L,
                weight_draws = 4L, noise_draws = 4L, monitor_draws = 4L, endpoint_draws = 4L,
                max_backtracks = 1L, lr = 1e5)

# Rejected proposals leave the initial factor in place. Check both projections
# and the paper's unmodified default for all priors.
for (prior in c("G", "GL", "GLT")) {
  for (bounds in list(c(-2, -1), c(-8, -7), c(-15, 3))) {
    control$logsd_bounds <- bounds
    fit <- gltbnn_fit(x, y, hidden = 2L, prior = prior, control = control, seed = 17L)
    values <- unlist(fit$engine$parameters$logsd)
    expected <- max(bounds[1L], min(log(0.01), bounds[2L]))
    stopifnot(!any(fit$engine$weight_steps$accepted), all(values == expected),
              all(values >= bounds[1L] & values <= bounds[2L]), is.finite(fit$engine$endpoint$value))
  }
}

# Extreme bounds can map to zero or infinity in double precision, including
# when the initializer itself would still lie inside the proposed interval.
for (bounds in list(c(-1001, -1000), c(999, 1000), c(-500, 3), c(-15, 500))) {
  control$logsd_bounds <- bounds
  msg <- tryCatch(gltbnn_fit(x, y, hidden = 2L, control = control), error = conditionMessage)
  stopifnot(is.character(msg), grepl("positive finite variances", msg, fixed = TRUE))
}
control$logsd_bounds <- c(-370, 350)
fit <- gltbnn_fit(x, y, hidden = 2L, prior = "G", control = control, seed = 17L)
stopifnot(all(is.finite(exp(2 * unlist(fit$engine$parameters$logsd)))),
          all(exp(2 * unlist(fit$engine$parameters$logsd)) > 0))
