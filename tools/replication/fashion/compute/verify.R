# Numerical equivalence checks for the binary minibatch extension.
# Run from the study root before estimating the comparison.
source("compute/full_fit.R")
source("compute/dropout.R")
engine <- fashion_engine()
base <- new.env(parent = globalenv())
for (f in list.files("package/R", full.names = TRUE)) sys.source(f, base)
same <- function(a, b, tolerance = 1e-12) {
  stopifnot(isTRUE(all.equal(a, b, tolerance = tolerance, check.attributes = FALSE)))
}
set.seed(91)
x <- matrix(rnorm(60), nrow = 15)
y <- rep(1:2, length.out = nrow(x))
w <- engine$.bnn_initialize(x, y, c(4L, 3L), 57L, 0L, "binary", 2L)
par <- list(mu = w, logsd = lapply(w, function(v) v * 0 - 4.6))
eps <- engine$.bnn_antithetic_draws(par, 4L)
rates <- lapply(w, function(v) v * 0 + 1.7)
r <- rep(seq_len(nrow(x)), 3L)
objective <- function(e, xx, yy, scale) {
  e$.bnn_weight_objective(par, xx, yy, scale, rates, eps, TRUE, "binary")
}
same(objective(engine, x, y, 1), objective(base, x, y, 1))
same(objective(engine, x, y, 3), objective(base, x[r, ], y[r], 1))
# Check reparameterized Bernoulli mean and log-SD derivatives against a
# central difference, keeping the same latent draws and minibatch scaling.
adjust <- function(z) lapply(z, function(m) matrix(rnorm(length(m)), nrow(m)))
direction <- list(mu = adjust(par$mu), logsd = adjust(par$logsd))
perturb <- function(h) {
  values <- Map(function(a, b) Map(function(v, d) v + h * d, a, b), par, direction)
  engine$.bnn_weight_objective(values, x, y, 3, rates, eps, FALSE, "binary")
}
analytic <- objective(engine, x, y, 3)$gradient
same((perturb(1e-6) - perturb(-1e-6)) / 2e-6,
     sum(unlist(Map(function(g, d) sum(g*d), analytic, c(direction$mu, direction$logsd)))),
     tolerance = 1e-6)
control <- engine$.gltbnn_control(list(outer = 2L, weight_steps = 2L))
zero <- lapply(c(w, w), function(v) v * 0)
one_step <- function(e, xx, yy, n_total) {
  set.seed(310)
  e$.bnn_adam_step(par, list(m = zero, v = zero), 1L, xx, yy, 1, rates, control,
                   "binary", n_total)
}
same(one_step(engine, x, y, length(r)), one_step(base, x[r, ], y[r], length(r)))
same(fashion_fnn_update(list(x = x, y = y), w, rep(list(seq_len(nrow(x))), 3L),
                      "Unpenalized", engine, 57L),
     base$.bnn_initialize(x, y, c(4L, 3L), 57L, 3L, "binary", 2L))
for (prior in c("G", "GL", "GLT")) {
  fit <- engine$.gltbnn_engine_fit(x, y, c(4L,3L), prior, control, 57L,
                                   outcome = "binary", n_classes = 2L,
                                   initial_weights = w,
                                   batches = fashion_batches(nrow(x), 7L, 4L, 123L))
  stopifnot(is.null(fit$state$noise), nrow(fit$weight_steps) == 4L,
            all(is.finite(fit$weight_steps$sample_objective)))
  pred <- fashion_predict(x, y-1L, engine, parameters = fit$parameters,
                        seed = 261L, draws = 11L, chunk_size = 4L)
  set.seed(261L)
  ref <- base$.gltbnn_engine_class_draws(fit, x, 11L, TRUE)
  same(pred$prob, ref)
  same(pred$true_logprob, log(ref[cbind(seq_len(nrow(x)), y)]))
}
# Both binary probabilities are checked against a separate logistic calculation,
# including large logits for which ordinary probabilities underflow.
eta <- c(-1000, -2.1, 0, 1.7, 1000)
truth <- c(1L, 0L, 1L, 1L, 0L)
single <- list(matrix(c(1, 0), nrow = 1))
pred <- fashion_predict(matrix(eta), truth, engine, weights = single, chunk_size = 1L)
same(pred$prob[, "pullover"], plogis(eta))
same(pred$prob[, "T-shirt/top"], plogis(-eta))
stable <- function(t) pmin(t, 0) - log1p(exp(-abs(t)))
same(pred$true_logprob, stable((2 * truth - 1) * eta))
same(pred$pred, as.integer(eta > 0))

# Fix the dropout masks and check the analytic gradient in a diagonal direction.
set.seed(415)
direction <- lapply(w, function(m) matrix(rnorm(length(m)), nrow(m)))
objective_dropout <- function(weights) {
  h <- x
  for (l in seq_along(weights)) {
    z <- cbind(h, 1) %*% t(weights[[l]])
    if (l < length(weights)) {
      h <- pmax(z, 0) * matrix(rbinom(length(z), 1L, .8) / .8, nrow(z))
    }
  }
  lp <- engine$.bnn_class_log_probs(z, "binary")
  -mean(lp[cbind(seq_along(y), y)])
}
set.seed(135)
g <- fashion_dropout_gradient(x, y, w, .2, engine)
score_after <- function(h) {
  set.seed(135)
  objective_dropout(Map(function(v, d) v+h*d, w, direction))
}
same((score_after(1e-6) - score_after(-1e-6)) / 2e-6,
     sum(unlist(Map(function(g, d) sum(g*d), g, direction))), tolerance = 1e-6)

for (seed in c(281225L, 291225L)) {
  batches <- fashion_batches(12000L, 1024L, 600L, seed)
  stopifnot(length(batches) == 600L, sum(lengths(batches)) == 600000L)
  for (epoch in 0:49) {
    ids <- unlist(batches[12L*epoch + seq_len(12L)], use.names = FALSE)
    stopifnot(identical(sort(ids), seq_len(12000L)))
  }
}
cat("Passed binary objective and gradient, Adam, dropout, initialization, all three priors, posterior prediction and full-coverage checks.\n")
