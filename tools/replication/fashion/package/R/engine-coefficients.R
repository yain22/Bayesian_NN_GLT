# Gauss-Legendre nodes and weights on [0,1] from the Jacobi matrix.
.bnn_scalar_rule <- local({
  n <- 32L
  j <- seq_len(n - 1L)
  J <- matrix(0, n, n)
  J[cbind(j, j + 1L)] <- j / sqrt(4 * j^2 - 1)
  ev <- eigen(J + t(J), symmetric = TRUE)
  list(nodes = (1 + ev$values) / 2, weights = ev$vectors[1, ]^2)
})

# q(lambda) is proportional to lambda^-1 exp(-A lambda - B/(2 lambda^2)).
# With m = log(B/A)/3 and s = log(lambda)-m, its log-scale density depends
# only on q = (A^2 B)^(1/3).  Its mode is s=0. Integrate both sides there,
# using shorter panels for very broad kernels and curvature scaling for narrow ones.
.bnn_scalar_locals <- function(A, B, d = 1L, normalizer_only = FALSE) {
  if (length(A) != length(B) || any(!is.finite(A) | A <= 0 | !is.finite(B) | B <= 0)) {
    stop("Scalar local kernel requires one positive finite A and B for each coefficient.")
  }
  log_a <- log(A)
  log_b <- log(B)
  m <- (log_b - log_a) / 3
  log_q <- (2 * log_a + log_b) / 3
  q <- exp(log_q)
  # The endpoints suppress the exponential tails by at least exp(-70) on
  # ordinary scales, including the first-moment integrand.
  upper <- log1p(80 / q + 14 / sqrt(q))
  lower <- -.5 * log1p(160 / q + 28 / sqrt(q))
  tiny <- q < 1e-300
  if (any(tiny)) {
    upper[tiny] <- log(80 + 14 * sqrt(q[tiny]) + q[tiny]) - log_q[tiny]
    lower[tiny] <- -.5 * (log(160 + 28 * sqrt(q[tiny]) + q[tiny]) - log_q[tiny])
  }
  if (any(!is.finite(upper) | !is.finite(lower))) stop("Local scale exceeds quadrature range.")
  segments <- pmax(1L, ceiling(pmax(-lower, upper) / 6))
  z <- raw_first <- raw_log <- raw_energy <- numeric(length(q))
  for (count in unique(segments)) {
    idx <- which(segments == count)
    Q <- q[idx]
    tiny_q <- tiny[idx]
    total <- first <- log_scale <- energy <- numeric(length(idx))
    for (width in list(lower[idx] / count, upper[idx] / count)) {
      for (panel in seq_len(count)) {
        for (j in seq_along(.bnn_scalar_rule$nodes)) {
          s <- width * ((panel - 1L) + .bnn_scalar_rule$nodes[j])
          cost <- Q * (expm1(s) + .5 * expm1(-2 * s))
          if (any(tiny_q)) {
            b <- log_q[idx[tiny_q]]
            u <- s[tiny_q]
            cost[tiny_q] <- exp(b + u) + .5 * exp(b - 2*u) - 1.5*Q[tiny_q]
          }
          small <- abs(s) < 1e-4
          if (any(small)) {
            a <- s[small]
            cost[small] <- (Q[small] * a) * a * (1.5 + a * (-.5 + a * (.375 - .125 * a)))
          }
          if (d != 1L) cost <- cost + (d - 1L) * s
          weight <- abs(width) * .bnn_scalar_rule$weights[j]
          mass <- weight * exp(-cost)
          total <- total + mass
          first_exponent <- s - cost
          if (any(tiny_q)) first_exponent[tiny_q] <- first_exponent[tiny_q] + log_q[idx[tiny_q]]
          first <- first + weight * exp(first_exponent)
          log_scale <- log_scale + mass * s
          energy <- energy + mass * cost
        }
      }
    }
    z[idx] <- total
    raw_first[idx] <- first
    raw_log[idx] <- log_scale
    raw_energy[idx] <- energy
  }
  # The d=2 grouped first moment needs only the d=1 log normalizer. Its
  # unused scalar moments can exceed double precision for subnormal rates.
  if (normalizer_only) return(log(z) - 1.5*q + (1-d)*m)
  zeta <- m + raw_log / z
  log_u <- m + log(raw_first / z)
  if (any(tiny)) log_u[tiny] <- log_u[tiny] - log_q[tiny]
  # Integration by parts yields B E[lambda^-2] = A E[lambda] exactly.
  u <- exp(log_u)
  log_phi <- log_a - log_b + log_u
  if (d != 1L) log_phi <- log_phi + log1p((d-1L)*exp(-log_a-log_u))
  phi <- exp(log_phi)
  entropy <- log(z) + zeta + raw_energy / z
  if (any(!is.finite(u) | u <= 0 | !is.finite(phi) | phi <= 0 | !is.finite(entropy))) {
    stop("Nonfinite scalar local moments.")
  }
  list(u = u, phi = phi, zeta = zeta, entropy = entropy, A = A, B = B,
       log_normalizer = log(z) - 1.5 * q + (1-d)*m)
}

# Scalar local factors in R matrix order (receiving unit, incoming coordinate).
# Keeping one vector per moment allows all coefficients in a layer to update together.
.bnn_local_ig <- function(a, b) {
  if (!is.finite(a) || a <= 0 || any(!is.finite(b) | b <= 0)) stop("Invalid local inverse-gamma factor.")
  list(a = a, b = b, phi = a / b, zeta = log(b) - digamma(a),
       entropy = a + log(b) + lgamma(a) - (a + 1) * digamma(a))
}

.bnn_coefficient_tail <- function(count, C, fixed_shape) {
  if (is.null(fixed_shape)) return(.bnn_tail_factor(count, C))
  eta <- log(fixed_shape)
  list(C = C, K = count, r = 1 / fixed_shape, eta_mean = eta,
       log_normalizer = .bnn_quad_tail_g(eta, count, C), log_prior_support = 0)
}

.bnn_coefficient_init <- function(count, prior, fixed_shape = NULL) {
  g <- list(geometry = "coefficient", M = count, tau = .bnn_ig(2, 2))
  if (prior == "GLT") {
    g$nu <- .bnn_local_ig(2, rep(2, count))
    start <- .bnn_scalar_locals(1, 1)
    g$local <- lapply(start, rep, length.out = count)
    g$fixed_shape <- fixed_shape
    g$tail <- .bnn_coefficient_tail(count, count * g$tau$zeta + sum(g$nu$zeta + g$nu$phi), fixed_shape)
  } else {
    g$aux <- .bnn_ig(1, 1)
    if (prior == "GL") {
      g$local <- .bnn_local_ig(2, rep(2, count))
      g$nu <- .bnn_local_ig(1, rep(1, count))
    }
  }
  g
}

.bnn_coefficient_terms <- function(g, prior) {
  M <- g$M
  if (prior == "GLT") {
    C <- M * g$tau$zeta + sum(g$nu$zeta + g$nu$phi)
    hyper <- -M * g$tau$zeta - sum(g$nu$zeta) - g$tau$phi * sum(g$local$u * g$nu$phi) +
      sum(g$local$entropy) + g$tail$log_normalizer - g$tail$log_prior_support -
      (C - g$tail$C) * g$tail$r - 2 * g$tau$zeta - sum(g$nu$zeta) - g$tau$phi +
      g$tau$entropy + sum(g$nu$entropy)
    return(list(precision = g$local$phi, logvar = 2 * g$local$zeta, hyper = hyper))
  }
  hyper <- .bnn_ig_log_prior(g$tau, .5, g$aux$phi, -g$aux$zeta) + g$tau$entropy +
    .bnn_ig_log_prior(g$aux, .5, M^2, log(M^2)) + g$aux$entropy
  if (prior == "G") {
    return(list(precision = rep(g$tau$phi, M), logvar = rep(g$tau$zeta, M), hyper = hyper))
  }
  hyper <- hyper + sum(.bnn_ig_log_prior(g$local, .5, g$nu$phi, -g$nu$zeta) +
    g$local$entropy + .bnn_ig_log_prior(g$nu, .5, 1, 0) + g$nu$entropy)
  list(precision = g$tau$phi * g$local$phi, logvar = g$tau$zeta + g$local$zeta, hyper = hyper)
}

.bnn_coefficient_update <- function(g, prior, moments, phi) {
  M <- g$M
  if (length(moments) != M) stop("Coefficient moments must cover each weight and bias.")
  if (prior == "GLT") {
    g$local <- .bnn_scalar_locals(g$tau$phi * g$nu$phi, phi * moments)
    g$tau <- .bnn_ig(1 + M * (1 + g$tail$r), 1 + sum(g$local$u * g$nu$phi))
    g$nu <- .bnn_local_ig(1 + g$tail$r, g$local$u * g$tau$phi + g$tail$r)
    g$tail <- .bnn_coefficient_tail(M, M * g$tau$zeta + sum(g$nu$zeta + g$nu$phi), g$fixed_shape)
  } else {
    local_precision <- 1
    if (prior == "GL") {
      g$local <- .bnn_local_ig(1, g$nu$phi + phi * moments * g$tau$phi / 2)
      local_precision <- g$local$phi
      g$nu <- .bnn_local_ig(1, 1 + local_precision)
    }
    g$tau <- .bnn_ig((M + 1) / 2, g$aux$phi + phi * sum(moments * local_precision) / 2)
    g$aux <- .bnn_ig(1, M^2 + g$tau$phi)
  }
  g
}
