# Reproduce the scalar GLT quadrature check in the supplement with gltbnn 0.2.0.
# In t = log(lambda), the h-th raw moment is Z_h / Z_0, where the
# integrand of Z_h is exp(h*t - A*exp(t) - B*exp(-2*t)/2).
production <- getFromNamespace(".bnn_scalar_locals", "gltbnn")

integrate_real_line <- function(f) {
  breaks <- c(-Inf, -8, -4, 0, 4, 8, Inf)
  pieces <- vapply(seq_len(length(breaks) - 1L), function(j) {
    integrate(f, breaks[j], breaks[j + 1L], subdivisions = 2000L,
              rel.tol = 1e-10, abs.tol = 1e-10 / (length(breaks) - 1L))$value
  }, numeric(1))
  sum(pieces)
}

# Center each integrand at its own mode and scale by the inverse square root
# of its log curvature before integrating with base R's adaptive quadrature.
moment_integral <- function(A, B, h) {
  log_a <- log(A)
  log_b <- log(B)
  derivative <- function(t) {
    log_h <- if (h == 0) -Inf else log(abs(h))
    shift <- max(log_h, log_a + t, log_b - 2 * t)
    sign(h) * exp(log_h - shift) - exp(log_a + t - shift) +
      exp(log_b - 2 * t - shift)
  }
  center <- (log_b - log_a) / 3
  radius <- 1
  while (derivative(center - radius) <= 0 || derivative(center + radius) >= 0) {
    radius <- 2 * radius
  }
  mode <- uniroot(derivative, center + c(-radius, radius), tol = 1e-12)$root
  left <- exp(log_a + mode)
  right <- exp(log_b - 2 * mode)
  scale <- 1 / sqrt(left + 2 * right)
  density <- function(z) {
    delta <- scale * z
    exponent <- h * delta - left * expm1(delta) - (right / 2) * expm1(-2 * delta)
    exponent[!is.finite(exponent)] <- -Inf
    exp(exponent)
  }
  mass <- integrate_real_line(density)
  log_z <- h * mode - left - right / 2 + log(scale) + log(mass)
  list(log_z = log_z, mode = mode, scale = scale, mass = mass, density = density)
}

reference <- function(A, B) {
  base <- moment_integral(A, B, 0)
  mean_lambda <- exp(moment_integral(A, B, 1)$log_z - base$log_z)
  mean_inverse_square <- exp(moment_integral(A, B, -2)$log_z - base$log_z)
  centered <- integrate_real_line(function(z) z * base$density(z))
  mean_log <- base$mode + base$scale * centered / base$mass
  entropy <- base$log_z + mean_log + A * mean_lambda + (B / 2) * mean_inverse_square
  c(mean = mean_lambda, precision = mean_inverse_square, mean_log = mean_log,
    entropy = entropy)
}

grid <- expand.grid(A = c(0.001, 1, 1000), B = c(0.0001, 1, 100))
errors <- t(vapply(seq_len(nrow(grid)), function(i) {
  A <- grid$A[i]
  B <- grid$B[i]
  ref <- reference(A, B)
  fit <- production(A, B)
  c(mean_relative = abs(fit$u / ref[["mean"]] - 1),
    precision_relative = abs(fit$phi / ref[["precision"]] - 1),
    mean_log_absolute = abs(fit$zeta - ref[["mean_log"]]),
    entropy_absolute = abs(fit$entropy - ref[["entropy"]]))
}, numeric(4)))
print(cbind(grid, errors), digits = 5, row.names = FALSE)
print(apply(errors, 2, max), digits = 8)
cat(sprintf("Initial precision at A = B = 1: %.10f\n", production(1, 1)$phi))
