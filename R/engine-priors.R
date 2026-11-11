# For X inverse-gamma with shape a and scale b, 1/X is gamma with rate b.
# phi is E[1/X] and zeta is E[log X] under the variational factor.
.bnn_ig <- function(a, b) {
  if (!is.finite(a) || !is.finite(b) || a <= 0 || b <= 0) {
    stop("Invalid inverse-gamma variational parameters.")
  }
  list(a = a, b = b, phi = a / b, zeta = log(b) - digamma(a),
       entropy = a + log(b) + lgamma(a) - (a + 1) * digamma(a))
}

.bnn_ig_log_prior <- function(q, a, rate_mean, rate_log) {
  a * rate_log - lgamma(a) - (a + 1) * q$zeta - rate_mean * q$phi
}

.bnn_tail_stat <- function(g) {
  length(g$local) * g$tau$zeta + sum(vapply(g$nu, function(x) x$zeta + x$phi, numeric(1)))
}

.bnn_init_groups <- function(K, d, prior) {
  g <- list(tau = .bnn_ig(2, 2))
  if (prior == "GLT") {
    g$nu <- rep(list(.bnn_ig(2, 2)), K)
    g$local <- rep(list(.bnn_local_factor(1, 1, d)), K)
    g$tail <- .bnn_tail_factor(K, .bnn_tail_stat(g))
  } else {
    g$aux <- .bnn_ig(1, 1)
    if (prior == "GL") {
      g$local <- rep(list(.bnn_ig(2, 2)), K)
      g$nu <- rep(list(.bnn_ig(1, 1)), K)
    }
  }
  g
}

# Per-unit E[lambda^-2], E[log(lambda^2)], and the layer's contribution
# from scale priors plus scale-factor entropies to the augmented-bound ELBO.
.bnn_prior_terms <- function(g, K, prior) {
  if (prior == "GLT") {
    nu_precision <- vapply(g$nu, `[[`, numeric(1), "phi")
    nu_log <- vapply(g$nu, `[[`, numeric(1), "zeta")
    local_mean <- vapply(g$local, `[[`, numeric(1), "u")
    C <- K * g$tau$zeta + sum(nu_log + nu_precision)
    hyper <- -K * g$tau$zeta - sum(nu_log) - g$tau$phi * sum(local_mean * nu_precision) +
      sum(vapply(g$local, `[[`, numeric(1), "entropy")) +
      g$tail$log_normalizer - g$tail$log_prior_support - (C - g$tail$C) * g$tail$r -
      2 * g$tau$zeta - sum(nu_log) - g$tau$phi + g$tau$entropy +
      sum(vapply(g$nu, `[[`, numeric(1), "entropy"))
    return(list(precision = vapply(g$local, `[[`, numeric(1), "phi"),
                logvar = 2 * vapply(g$local, `[[`, numeric(1), "zeta"), hyper = hyper))
  }
  hyper <- .bnn_ig_log_prior(g$tau, .5, g$aux$phi, -g$aux$zeta) + g$tau$entropy +
    .bnn_ig_log_prior(g$aux, .5, K^2, log(K^2)) + g$aux$entropy
  precision <- rep(g$tau$phi, K)
  logvar <- rep(g$tau$zeta, K)
  if (prior == "GL") {
    for (k in seq_len(K)) {
      q <- g$local[[k]]
      aux <- g$nu[[k]]
      precision[k] <- precision[k] * q$phi
      logvar[k] <- logvar[k] + q$zeta
      hyper <- hyper + .bnn_ig_log_prior(q, .5, aux$phi, -aux$zeta) + q$entropy +
        .bnn_ig_log_prior(aux, .5, 1, 0) + aux$entropy
    }
  }
  list(precision = precision, logvar = logvar, hyper = hyper)
}

.bnn_update_groups <- function(g, K, d, prior, moments, phi) {
  if (prior == "GLT") {
    for (k in seq_len(K)) {
      g$local[[k]] <- .bnn_local_factor(g$tau$phi * g$nu[[k]]$phi, phi * moments[k], d)
    }
    local_rates <- vapply(seq_len(K), function(k) g$local[[k]]$u * g$nu[[k]]$phi, numeric(1))
    g$tau <- .bnn_ig(1 + K * (1 + g$tail$r), 1 + sum(local_rates))
    for (k in seq_len(K)) {
      g$nu[[k]] <- .bnn_ig(1 + g$tail$r, g$local[[k]]$u * g$tau$phi + g$tail$r)
    }
    g$tail <- .bnn_tail_factor(K, .bnn_tail_stat(g))
  } else {
    local_precision <- rep(1, K)
    if (prior == "GL") {
      for (k in seq_len(K)) {
        g$local[[k]] <- .bnn_ig((d + 1) / 2, g$nu[[k]]$phi + phi * moments[k] * g$tau$phi / 2)
        local_precision[k] <- g$local[[k]]$phi
        g$nu[[k]] <- .bnn_ig(1, 1 + local_precision[k])
      }
    }
    g$tau <- .bnn_ig((K * d + 1) / 2, g$aux$phi + phi * sum(moments * local_precision) / 2)
    g$aux <- .bnn_ig(1, K^2 + g$tau$phi)
  }
  g
}
