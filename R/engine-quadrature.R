# Deterministic quadrature for GLT scale factors, on the log scale.
# The local density has kernel lambda^(-d) exp(-A*lambda-B/(2*lambda^2)).
# Integration centers and scales each moment around its own mode.
.bnn_quad_positive <- function(x, label) {
  if (!length(x) || any(!is.finite(x)) || any(x <= 0))
    stop(label, " must be finite and positive.")
  invisible(x)
}

.bnn_quad_quadrature_settings <- function(rel_tol, subdivisions) {
  if (length(rel_tol) != 1L || !is.finite(rel_tol) || rel_tol <= 0 || rel_tol >= 1 ||
      length(subdivisions) != 1L || !is.finite(subdivisions) || subdivisions < 20 ||
      subdivisions != as.integer(subdivisions)) stop("Invalid quadrature settings.")
  invisible(NULL)
}

.bnn_quad_integrate <- function(f, lower, upper, rel_tol, subdivisions,
                                 extra = numeric()) {
  breaks <- sort(unique(c(lower, upper,
    c(-8, -4, 0, 4, 8, extra)[c(-8, -4, 0, 4, 8, extra) > lower &
                              c(-8, -4, 0, 4, 8, extra) < upper])))
  values <- errors <- numeric(length(breaks) - 1L)
  for (j in seq_along(values)) {
    ans <- stats::integrate(f, breaks[j], breaks[j + 1L], rel.tol = rel_tol,
      abs.tol = rel_tol / length(values), subdivisions = subdivisions,
      stop.on.error = TRUE)
    values[j] <- ans$value
    errors[j] <- ans$abs.error
  }
  list(value = sum(values), abs_error = sum(errors))
}

.bnn_quad_local_integral <- function(A, B, d, power, rel_tol, subdivisions) {
  coefficient <- 1 - d + power
  la <- log(A); lb <- log(B)
  derivative <- function(t) {
    z <- max(if (coefficient == 0) -Inf else log(abs(coefficient)), la + t, lb - 2 * t)
    term <- if (coefficient == 0) 0 else sign(coefficient) * exp(log(abs(coefficient)) - z)
    term - exp(la + t - z) + exp(lb - 2 * t - z)
  }
  midpoint <- (lb - la) / 3
  radius <- 1
  for (j in 1:60) {
    lower <- midpoint - radius; upper <- midpoint + radius
    if (derivative(lower) > 0 && derivative(upper) < 0) break
    radius <- radius * 2
  }
  if (!(derivative(lower) > 0 && derivative(upper) < 0))
    stop("Could not bracket a local quadrature mode.")
  mode <- stats::uniroot(derivative, c(lower, upper), tol = 1e-12)$root
  at <- exp(la + mode); bt <- exp(lb - 2 * mode)
  scale <- 1 / sqrt(at + 2 * bt)
  .bnn_quad_positive(scale, "Local quadrature scale")
  log_height <- coefficient * mode - at - bt / 2
  density <- function(z) {
    delta <- scale * z
    exponent <- coefficient * delta - at * expm1(delta) - (bt / 2) * expm1(-2 * delta)
    exponent[!is.finite(exponent)] <- -Inf
    exp(exponent)
  }
  integral <- .bnn_quad_integrate(density, -Inf, Inf, rel_tol, subdivisions)
  .bnn_quad_positive(integral$value, "Local quadrature integral")
  list(log_value = log_height + log(scale) + log(integral$value),
       mode = mode, scale = scale, density = density, integral = integral)
}

.bnn_local_factor <- function(A, B, d, rel_tol = 1e-10,
                                     subdivisions = 2000L) {
  .bnn_quad_quadrature_settings(rel_tol, subdivisions)
  .bnn_quad_positive(c(A, B), "Local kernel parameters")
  if (length(A) != 1L || length(B) != 1L || length(d) != 1L ||
      !is.finite(d) || d < 1 || d != as.integer(d)) stop("Invalid local group dimension.")
  base <- .bnn_quad_local_integral(A, B, d, 0, rel_tol, subdivisions)
  first <- .bnn_quad_local_integral(A, B, d, 1, rel_tol, subdivisions)
  inverse <- .bnn_quad_local_integral(A, B, d, -2, rel_tol, subdivisions)
  centered <- .bnn_quad_integrate(function(z) z * base$density(z),
    -Inf, Inf, rel_tol, subdivisions)
  u <- exp(first$log_value - base$log_value)
  phi <- exp(inverse$log_value - base$log_value)
  zeta <- base$mode + base$scale * centered$value / base$integral$value
  .bnn_quad_positive(c(u, phi), "Local quadrature moments")
  entropy <- base$log_value + d * zeta + A * u + (B / 2) * phi
  list(log_normalizer = base$log_value, u = u, phi = phi, zeta = zeta,
       entropy = entropy, A = A, B = B, d = d,
       identity_residual = B * phi - A * u - (d - 1),
       estimated_relative_integration_error = max(
         base$integral$abs_error / base$integral$value,
         first$integral$abs_error / first$integral$value,
         inverse$integral$abs_error / inverse$integral$value))
}

# lgamma(r) = lgamma(1+r) + eta avoids overflow when r=exp(-eta)
# underflows. The numerical integral includes the normalized Gaussian density.
.bnn_quad_tail_g <- function(eta, K, C) {
  r <- exp(-eta)
  -(K * eta + C) * r - K * (lgamma(1 + r) + eta) - lgamma(1 + K * r)
}

.bnn_tail_factor <- function(K, C, mu = 0, rho = 1,
                                    rel_tol = 1e-10, subdivisions = 2000L) {
  .bnn_quad_quadrature_settings(rel_tol, subdivisions)
  .bnn_quad_positive(c(K, rho), "Tail dimensions/reference scale")
  if (length(K) != 1L || K != as.integer(K) || length(C) != 1L ||
      !is.finite(C) || length(mu) != 1L || !is.finite(mu) || length(rho) != 1L)
    stop("Invalid tail kernel parameters.")
  boundary <- -log(2)
  log_kernel <- function(eta) stats::dnorm(eta, mu, rho, log = TRUE) +
    .bnn_quad_tail_g(eta, K, C)
  derivative <- function(eta) {
    r <- exp(-eta)
    gp <- r * (K * eta + C - K + K * digamma(1 + r) +
                 K * digamma(1 + K * r)) - K
    -(eta - mu) / rho^2 + gp
  }
  upper <- max(boundary + 8, mu + 12 * rho, log1p(abs(C) + K) + 8)
  for (iteration in 1:20) {
    grid <- seq(boundary, upper, length.out = 513L)
    heights <- log_kernel(grid)
    interior <- which(heights[2:512] >= heights[1:511] &
                       heights[2:512] >= heights[3:513]) + 1L
    candidates <- c(boundary, upper)
    if (length(interior)) for (j in interior) candidates <- c(candidates,
      stats::optimize(log_kernel, c(grid[j - 1L], grid[j + 1L]), maximum = TRUE,
               tol = 1e-11)$maximum)
    mode <- candidates[which.max(log_kernel(candidates))]
    peak <- log_kernel(mode)
    if (log_kernel(upper) < peak - 50 && derivative(upper) < 0) break
    upper <- upper + max(8 * rho, upper - boundary)
  }
  if (!(log_kernel(upper) < peak - 50 && derivative(upper) < 0))
    stop("Could not stabilize the tail quadrature range.")
  rmode <- exp(-mode)
  gp <- derivative(mode) + (mode - mu) / rho^2
  curvature <- 1 / rho^2 + gp + K - K * rmode +
    K * rmode^2 * trigamma(1 + rmode) + K^2 * rmode^2 * trigamma(1 + K * rmode)
  if (mode == boundary) {
    scale <- 1 / (abs(derivative(boundary)) + 1 / rho)
  } else {
    scale <- if (is.finite(curvature) && curvature > 0) 1 / sqrt(curvature) else rho
  }
  .bnn_quad_positive(scale, "Tail quadrature scale")
  lower_z <- (boundary - mode) / scale
  other_modes <- (candidates - mode) / scale
  breaks <- as.vector(outer(other_modes, c(-1, 0, 1), "+"))
  density <- function(z) {
    eta <- mode + scale * z
    exponent <- log_kernel(eta) - peak
    exponent[!is.finite(exponent)] <- -Inf
    exp(exponent)
  }
  integral <- .bnn_quad_integrate(density, lower_z, Inf,
    rel_tol, subdivisions, extra = breaks)
  .bnn_quad_positive(integral$value, "Tail quadrature integral")
  expectation <- function(f) {
    ans <- .bnn_quad_integrate(function(z) {
      eta <- mode + scale * z
      w <- density(z)
      result <- numeric(length(z))
      positive <- w > 0
      result[positive] <- f(eta[positive]) * w[positive]
      result
    }, lower_z, Inf, rel_tol, subdivisions, extra = breaks)
    ans$value / integral$value
  }
  logZ <- peak + log(scale) + log(integral$value)
  log_support <- stats::pnorm((mu - boundary) / rho, log.p = TRUE)
  r <- expectation(function(eta) exp(-eta))
  normal_log <- expectation(function(eta) stats::dnorm(eta, mu, rho, log = TRUE))
  g_mean <- expectation(function(eta) .bnn_quad_tail_g(eta, K, C))
  list(log_normalizer = logZ, r = r, eta_mean = expectation(identity),
       log_prior_support = log_support, expected_log_reference = normal_log,
       expected_log_factor = g_mean, entropy_eta = logZ - normal_log - g_mean,
       C = C, K = K, mu = mu, rho = rho, mode = mode,
       estimated_relative_integration_error = integral$abs_error / integral$value)
}
