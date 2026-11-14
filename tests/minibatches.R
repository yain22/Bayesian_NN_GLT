library(gltbnn)

# Reference predictions from the independent minibatch study driver.
set.seed(87)
x <- matrix(rnorm(43 * 3), nrow = 43)
seed <- 510125L
control <- list(outer = 3L, weight_steps = 4L, init_steps = 7L, batch_size = 12L)
reference <- list(continuous = c(-0.335015278308, -0.443036540363,
                                  -0.381750879391, -0.270576759389),
                  binary = c(0.344229584070, 0.321904720976,
                             0.335601683895, 0.341213149013))
saved_rng <- .Random.seed
for (family in c("continuous", "binary", "categorical")) {
  y <- switch(family, continuous = x[, 1] + x[, 2]^2,
              binary = rep(0:1, length.out = nrow(x)),
              categorical = rep(0:2, length.out = nrow(x)))
  fit <- gltbnn_fit(x, y, hidden = c(7L, 6L), prior = "GLT", control = control,
                    seed = seed, outcome = family)
  stopifnot(nrow(fit$engine$weight_steps) == 12L,
            all(is.na(fit$engine$trace$elbo)), is.na(fit$engine$endpoint$value))
  p <- predict(fit, x[1:4, , drop = FALSE],
               type = if (family == "continuous") "mean" else "prob",
               draws = 8L, seed = seed + 5000L)
  if (family != "continuous") stopifnot(all(abs(rowSums(p) - 1) < 1e-12))
  if (family %in% names(reference)) {
    actual <- if (family == "binary") p[, 2] else p
    stopifnot(max(abs(actual - reference[[family]])) < 1e-8)
  }
}
stopifnot(identical(saved_rng, .Random.seed))
