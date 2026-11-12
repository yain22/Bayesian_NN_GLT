library(gltbnn)

internal <- function(name) getFromNamespace(name, "gltbnn")
init <- internal(".bnn_init_groups")
update <- internal(".bnn_update_groups")
terms <- internal(".bnn_prior_terms")

# Treat the four entries of a 2-by-2 hidden matrix as four separate scalar
# groups. This independent route uses the older adaptive local integrations.
moments <- matrix(c(.11, .23, 1.8, .42), 2L, 2L)
for (prior in c("G", "GL", "GLT")) {
  g <- init(4L, 1L, prior, "coefficient")
  ref <- init(4L, 1L, prior, "unit")
  for (i in 1:3) {
    g <- update(g, 4L, 1L, prior, moments, 1.3)
    ref <- update(ref, 4L, 1L, prior, as.numeric(moments), 1.3)
    est <- terms(g, 4L, prior)
    target <- terms(ref, 4L, prior)
    for (key in names(target)) stopifnot(max(abs(est[[key]] - target[[key]])) < 1e-8)
    if (prior == "G") stopifnot(length(unique(est$precision)) == 1L)
    if (prior != "G") stopifnot(est$precision[1] > est$precision[3])
    if (prior == "GLT") {
      stopifnot(max(abs(g$local$B * g$local$phi - g$local$A * g$local$u)) < 1e-9,
                g$tail$K == length(moments))
    }
  }
}

# Independent adaptive integration on the logarithmic scale checks the vector
# quadrature, including its two tails and the entropy Jacobian. Compute the
# inverse moment by direct integration rather than using its moment identity.
scalar <- internal(".bnn_scalar_locals")
for (q in c(1e-13, .01, 1, 100, 1e6)) {
  m <- .7
  a <- q * exp(-m)
  b <- q * exp(2 * m)
  actual <- scalar(a, b)
  bounds <- c(-.5 * log1p(240/q + 32/sqrt(q)), 0, log1p(120/q + 16/sqrt(q)))
  cost <- function(s) q * (expm1(s) + .5 * expm1(-2*s))
  integrate_moment <- function(fun) {
    sum(vapply(1:2, function(i) {
      stats::integrate(function(s) fun(s) * exp(-cost(s)), bounds[i], bounds[i+1L],
                       abs.tol = 0, rel.tol = 1e-11)$value
    }, numeric(1)))
  }
  Z <- integrate_moment(function(s) 1)
  first <- integrate_moment(function(s) exp(m+s)) / Z
  inverse <- integrate_moment(function(s) exp(-2*(m+s))) / Z
  log_scale <- m + integrate_moment(identity) / Z
  entropy <- log(Z) + log_scale + integrate_moment(cost) / Z
  stopifnot(abs(actual$u/first - 1) < 1e-9, abs(actual$phi/inverse - 1) < 1e-9,
            abs(actual$zeta - log_scale) < 1e-9, abs(actual$entropy - entropy) < 1e-8)
}

# The current unit's bias is its final incoming coordinate. Distinct
# precisions in the two columns detect rowwise broadcasting in the optimizer.
x <- matrix(c(-1, -.3, .5, 1), ncol = 1L)
y <- c(0, 1, .3, .8)
p <- list(mu = list(rbind(c(.3, 1.4), c(.7, 1.8)), matrix(c(.2, -.3, .1), nrow=1L)))
p$logsd <- lapply(p$mu, function(w) w*0 - 3)
set.seed(12L)
ep <- internal(".bnn_antithetic_draws")(p, 4L)
rates <- list(c(.5, .8, 5, 9), 1.7)
objective <- internal(".bnn_weight_objective")
gradient <- objective(p, x, y, 1.2, rates, ep, TRUE)$gradient
for (i in seq_along(p)) {
  for (j in seq_along(p[[i]][[1L]])) {
    plus <- minus <- p
    plus[[i]][[1L]][j] <- plus[[i]][[1L]][j] + 1e-6
    minus[[i]][[1L]][j] <- minus[[i]][[1L]][j] - 1e-6
    slope <- (objective(plus, x, y, 1.2, rates, ep) - objective(minus, x, y, 1.2, rates, ep))/2e-6
    k <- if (i == 1L) 1L else 3L
    stopifnot(abs(gradient[[k]][j] - slope) < 1e-7)
  }
}

# The internal application protocol scales only the batch likelihood, pools
# the cycle's rows for the variance update, and fixes the tail for ablations.
engine_fit <- internal(".gltbnn_engine_fit")
ctl <- internal(".gltbnn_control")(list(init_steps=1L, outer=1L, weight_steps=2L,
  weight_draws=2L, noise_draws=4L, monitor_draws=4L, endpoint_draws=4L))
batches <- list(c(1L, 3L), 4L)
for (xi in c(1, 2)) {
  fit <- engine_fit(x, y, 2L, "GLT", ctl, 11L, fixed_shape=xi, batches=batches)
  layer <- fit$state$g[[1]]
  stopifnot(layer$tail$r == 1/xi, layer$tail$eta_mean == log(xi), layer$tail$K == 4L,
            all(fit$trace$accepted + fit$trace$rejected == c(0L,2L)),
            is.finite(fit$trace$pooled_batch_rmse[2L]), fit$state$noise$a == (length(y)+4L)/2,
            all(is.na(fit$trace$elbo)), is.null(fit$endpoint$expected_sse))
}

# With n/b integral, n/b times the batch likelihood is exactly the likelihood
# of n rows obtained by repeating that batch. Reconstruct both Adam steps with
# this full-length design and independently calculate the pooled variance update.
W <- internal(".bnn_initialize")(x, y, 2L, 11L, 1L)
p <- list(mu = W, logsd = lapply(W, function(w) w*0 + log(.01)))
zero <- lapply(c(p$mu, p$logsd), function(w) w*0)
adam <- list(m = zero, v = zero)
pr <- terms(init(4L, 1L, "GLT", "coefficient", 2), 4L, "GLT")$precision
rates <- list(3 * pr, 1)
set.seed(12L)
for (t in seq_along(batches)) {
  ids <- rep(batches[[t]], nrow(x) / length(batches[[t]]))
  step <- internal(".bnn_adam_step")(p, adam, t, x[ids,,drop=FALSE], y[ids], 3, rates, ctl)
  p <- step$parameters
  adam <- step$adam
}
stopifnot(max(abs(unlist(p) - unlist(fit$parameters))) < 1e-12)
ep <- internal(".bnn_antithetic_draws")(p, ctl$noise_draws)
ids <- unlist(batches, use.names = FALSE)
sse <- mean(vapply(ep, function(z) {
  w <- Map(function(m,s,e) m + exp(s)*e, p$mu, p$logsd, z)
  h <- pmax(cbind(x[ids,,drop=FALSE],1) %*% t(w[[1]]), 0)
  sum((y[ids] - cbind(h,1) %*% t(w[[2]]))^2)
}, numeric(1))) * nrow(x) / length(ids)
penalty <- sum((p$mu[[1]]^2 + exp(2*p$logsd[[1]])) * pr)
stopifnot(abs(fit$state$noise$b - (sse + penalty)/2) < 1e-10,
          abs(fit$trace$pooled_batch_rmse[2L] - sqrt(sse/nrow(x))) < 1e-10)
