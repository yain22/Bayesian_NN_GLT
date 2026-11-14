# Numerical equivalence checks for the categorical minibatch extension.
# Run from the study root before estimating the comparison.
source("compute/full_fit.R")
source("compute/dropout.R")
engine <- mnist_engine()
base <- new.env(parent = globalenv())
for (f in list.files("package/R", full.names = TRUE)) sys.source(f, base)
same <- function(a, b, tolerance = 1e-12) {
  stopifnot(isTRUE(all.equal(a, b, tolerance = tolerance, check.attributes = FALSE)))
}
set.seed(91)
x <- matrix(rnorm(60), nrow = 15)
y <- rep(1:10, length.out = nrow(x))
w <- engine$.bnn_initialize(x, y, c(4L, 3L), 57L, 0L, "categorical", 10L)
par <- list(mu = w, logsd = lapply(w, function(v) v * 0 - 4.6))
eps <- engine$.bnn_antithetic_draws(par, 4L)
rates <- lapply(w, function(v) v * 0 + 1.7)
r <- rep(seq_len(nrow(x)), 3L)
objective <- function(e, xx, yy, scale) {
  e$.bnn_weight_objective(par, xx, yy, scale, rates, eps, TRUE, "categorical")
}
same(objective(engine, x, y, 1), objective(base, x, y, 1))
same(objective(engine, x, y, 3), objective(base, x[r, ], y[r], 1))
control <- engine$.gltbnn_control(list(outer = 2L, weight_steps = 2L))
zero <- lapply(c(w, w), function(v) v * 0)
one_step <- function(e, xx, yy, n_total) {
  set.seed(310)
  e$.bnn_adam_step(par, list(m = zero, v = zero), 1L, xx, yy, 1, rates, control,
                   "categorical", n_total)
}
same(one_step(engine, x, y, length(r)), one_step(base, x[r, ], y[r], length(r)))
same(mnist_fnn_update(list(x = x, y = y), w, rep(list(seq_len(nrow(x))), 3L),
                      "Unpenalized", engine, 57L),
     base$.bnn_initialize(x, y, c(4L, 3L), 57L, 3L, "categorical", 10L))
for (prior in c("G", "GL", "GLT")) {
  fit <- engine$.gltbnn_engine_fit(x, y, c(4L,3L), prior, control, 57L,
                                   outcome = "categorical", n_classes = 10L,
                                   initial_weights = w,
                                   batches = mnist_batches(nrow(x), 7L, 4L, 123L))
  stopifnot(is.null(fit$state$noise), nrow(fit$weight_steps) == 4L,
            all(is.finite(fit$weight_steps$sample_objective)))
  pred <- mnist_predict(x, y-1L, engine, parameters = fit$parameters,
                        seed = 261L, draws = 11L, chunk_size = 4L)
  set.seed(261L)
  ref <- base$.gltbnn_engine_class_draws(fit, x, 11L, TRUE)
  same(pred$prob, ref)
  same(pred$true_logprob, log(ref[cbind(seq_len(nrow(x)), y)]))
}
for (seed in c(281225L, 291225L)) {
  batches <- mnist_batches(60000L, 1024L, 600L, seed)
  stopifnot(length(batches) == 600L, sum(lengths(batches)) == 610240)
  for (epoch in 0:9) {
    ids <- unlist(batches[59L*epoch + seq_len(59L)], use.names = FALSE)
    stopifnot(identical(sort(ids), seq_len(60000L)))
  }
}
cat("Passed objective, gradient, Adam, initialization, all three categorical priors, predictive draws and full-schedule coverage checks.\n")
