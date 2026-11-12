library(gltbnn)

# Use elementary likelihoods and an explicit small network as an independent
# check on the orientation, signs and reparameterization gradients of the engine.
log_probs <- getFromNamespace(".bnn_class_log_probs", "gltbnn")
class_score <- getFromNamespace(".bnn_class_score", "gltbnn")
objective <- getFromNamespace(".bnn_weight_objective", "gltbnn")
antithetic <- getFromNamespace(".bnn_antithetic_draws", "gltbnn")
elbo <- getFromNamespace(".bnn_elbo", "gltbnn")
prior_terms <- getFromNamespace(".bnn_prior_terms", "gltbnn")

extreme <- rbind(c(1000, 1001), c(-1000, -1001), c(0, 0))
lp <- log_probs(extreme, "categorical")
stopifnot(all(is.finite(lp)), max(abs(rowSums(exp(lp)) - 1)) < 1e-14,
          abs(lp[1, 1] + log1p(exp(1))) < 1e-12, lp[2, 1] == -1000)
lp <- log_probs(matrix(c(40, 0), nrow = 1L), "categorical")
stopifnot(abs(lp[1, 1] / (-2 * exp(-40)) - 1) < 1e-14,
          abs(class_score(lp, 1L, "categorical")[1, 1] / (2 * exp(-40)) - 1) < 1e-14)
lp <- log_probs(matrix(c(-1000, 1000), ncol = 1L), "binary")
stopifnot(all(is.finite(lp)), lp[1, 2] == -1000, lp[2, 1] == -1000,
          identical(as.vector(class_score(lp, c(2L, 1L), "binary")), c(1, -1)))

x <- matrix(c(-1, 0, 0.4, 1.2), ncol = 1L)
for (outcome in c("binary", "categorical")) {
  binary <- outcome == "binary"
  y <- if (binary) c(1L, 2L, 2L, 1L) else c(1L, 2L, 1L, 3L)
  output_width <- max(y) - 1L
  mu <- list(rbind(c(.6, 1.2), c(-.3, -.1)),
             matrix(seq_len(3L * output_width) / 7, nrow = output_width))
  parameters <- list(mu = mu, logsd = lapply(mu, function(w) w * 0 - 3.7))
  set.seed(421L)
  epsilon <- antithetic(parameters, 4L)
  rates <- list(c(1.2, .8), .7)
  expected_loglik <- function(p, z, design = x) {
    w <- Map(function(m, s, e) m + exp(s) * e, p$mu, p$logsd, z)
    h <- pmax(cbind(design, 1) %*% t(w[[1L]]), 0)
    eta <- cbind(h, 1) %*% t(w[[2L]])
    if (binary) {
      return(sum(stats::dbinom(y - 1L, size = 1, prob = stats::plogis(eta), log = TRUE)))
    }
    unnormalized <- cbind(exp(eta), 1)
    sum(log(unnormalized[cbind(seq_along(y), y)] / rowSums(unnormalized)))
  }
  independent <- function(p) {
    moments <- Map(function(m, s) m^2 + exp(2*s), p$mu, p$logsd)
    sum(unlist(p$logsd)) - sum(unlist(Map(`*`, moments, rates))) / 2 +
      mean(vapply(epsilon, function(z) expected_loglik(p, z), numeric(1)))
  }
  engine <- objective(parameters, x, y, 1, rates, epsilon, TRUE, outcome)
  stopifnot(abs(independent(parameters) - engine$value) < 1e-12)
  gradient <- unlist(engine$gradient, use.names = FALSE)
  pos <- 0L
  step <- 1e-6
  for (part in c("mu", "logsd")) {
    for (l in seq_along(mu)) {
      for (j in seq_along(mu[[l]])) {
        plus <- minus <- parameters
        plus[[part]][[l]][j] <- plus[[part]][[l]][j] + step
        minus[[part]][[l]][j] <- minus[[part]][[l]][j] - step
        finite_difference <- (independent(plus) - independent(minus)) / (2 * step)
        pos <- pos + 1L
        stopifnot(abs(gradient[pos] - finite_difference) < 1e-7)
      }
    }
  }

  for (prior in c("G", "GL", "GLT")) {
    control <- list(init_steps = 2L, outer = 1L, weight_steps = 1L, weight_draws = 2L,
                    monitor_draws = 4L, endpoint_draws = 4L)
    fit <- gltbnn_fit(x, y, hidden = 2L, prior = prior, outcome = outcome,
                      control = control, seed = 12L)$engine
    stopifnot(is.null(fit$state$noise), is.null(fit$endpoint$expected_sse),
              all(is.finite(fit$trace$elbo)), !"noise_precision" %in% names(fit$trace),
              fit$state$u$a == (3L * output_width + 1L) / 2,
              abs(fit$endpoint$value - sum(fit$endpoint$components)) < 1e-10)

    # Check the monitor using the optimizer's actual scaled training design.
    design <- scale(x)
    eps <- antithetic(fit$parameters, 4L)
    sampled <- vapply(eps, function(z) expected_loglik(fit$parameters, z, design), numeric(1))
    pairs <- (sampled[1:2] + sampled[3:4]) / 2
    terms <- list(prior_terms(fit$state$g[[1]], 2L, prior))
    chk <- elbo(fit$state, fit$parameters, design, y, terms, eps, outcome = outcome)
    stopifnot(abs(chk$expected_loglik - mean(sampled)) < 1e-10,
              abs(chk$mcse - stats::sd(pairs) / sqrt(2)) < 1e-10)
  }
}
