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
                                 extra = numeric(), roundoff_tol = rel_tol) {
  breaks <- sort(unique(c(lower, upper,
    c(-8, -4, 0, 4, 8, extra)[c(-8, -4, 0, 4, 8, extra) > lower &
                              c(-8, -4, 0, 4, 8, extra) < upper])))
  values <- errors <- numeric(length(breaks) - 1L)
  retried <- FALSE
  for (j in seq_along(values)) {
    ans <- stats::integrate(f, breaks[j], breaks[j + 1L], rel.tol = rel_tol,
      abs.tol = rel_tol / length(values), subdivisions = subdivisions,
      stop.on.error = FALSE)
    if (ans$message != "OK" && roundoff_tol > rel_tol) {
      retried <- TRUE
      ans <- stats::integrate(f, breaks[j], breaks[j + 1L], rel.tol = roundoff_tol,
        abs.tol = roundoff_tol / length(values), subdivisions = subdivisions,
        stop.on.error = FALSE)
    }
    if (ans$message != "OK") {
      condition <- simpleError(ans$message)
      class(condition) <- c("gltbnn_quadrature_error",class(condition))
      stop(condition)
    }
    values[j] <- ans$value
    errors[j] <- ans$abs.error
  }
  list(value = sum(values), abs_error = sum(errors), retried = retried)
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
  remainder <- function(x) {
    ans <- expm1(x) - x
    small <- abs(x) < 1e-3
    t <- x[small]
    ans[small] <- t^2 * (.5 + t*(1/6 + t*(1/24 + t/120)))
    ans
  }
  offset <- function(delta) {
    exponent <- (coefficient-at+bt)*delta - at*remainder(delta) - bt/2*remainder(-2*delta)
    exponent[!is.finite(exponent)] <- -Inf
    exponent
  }
  density <- function(z) exp(offset(scale*z))
  integral <- .bnn_quad_integrate(density, -Inf, Inf, rel_tol, subdivisions)
  .bnn_quad_positive(integral$value, "Local quadrature integral")
  list(log_value = log_height + log(scale) + log(integral$value),
       mode = mode, scale = scale, density = density, offset = offset, integral = integral)
}

.bnn_local_factor <- function(A, B, d, rel_tol = 1e-10,
                                     subdivisions = 2000L) {
  .bnn_quad_quadrature_settings(rel_tol, subdivisions)
  .bnn_quad_positive(c(A, B), "Local kernel parameters")
  if (length(A) != 1L || length(B) != 1L || length(d) != 1L ||
      !is.finite(d) || d < 1 || d != as.integer(d)) stop("Invalid local group dimension.")
  log_q <- (2 * log(A) + log(B)) / 3
  broad <- log_q < log(1e-3)
  if ((broad && d == 1L) || log_q > log(1e4) + 2 * log(d)) {
    # The scalar rule resolves a broad d=1 plateau. In the concentrated
    # regime, the (1-d)s tilt shifts the mode by less than one hundredth of
    # a kernel SD; direct moments preserve the entropy on the lambda scale.
    ans <- .bnn_scalar_locals(A, B, d)
    ans$d <- d
    ans$identity_residual <- B * ans$phi - A * ans$u - (d - 1)
    ans$estimated_relative_integration_error <- NA_real_
    return(ans)
  }
  base <- .bnn_quad_local_integral(A, B, d, 0, rel_tol, subdivisions)
  first <- if (d == 2L && broad) {
    # The numerator of E[lambda] has the d=1 kernel. Its broad log-scale
    # plateau is resolved by the scalar rule's split panels.
    list(log_value = .bnn_scalar_locals(A,B,normalizer_only=TRUE))
  } else {
    .bnn_quad_local_integral(A, B, d, 1, rel_tol, subdivisions)
  }
  inverse <- .bnn_quad_local_integral(A, B, d, -2, rel_tol, subdivisions)
  centered <- .bnn_quad_integrate(function(z) z * base$density(z),
    -Inf, Inf, rel_tol, subdivisions)
  # Express moment ratios relative to the denominator mode before adding
  # their large log heights. This avoids subtracting nearly equal heights.
  log_ratio <- function(numerator, power) {
    if (is.null(numerator$integral)) return(numerator$log_value - base$log_value)
    base$offset(numerator$mode - base$mode) + power*numerator$mode +
      log(numerator$scale / base$scale) + log(numerator$integral$value / base$integral$value)
  }
  u <- exp(log_ratio(first,1))
  phi <- exp(log_ratio(inverse,-2))
  zeta <- base$mode + base$scale * centered$value / base$integral$value
  .bnn_quad_positive(c(u, phi), "Local quadrature moments")
  entropy_cost <- .bnn_quad_integrate(function(z) {
    h <- base$offset(base$scale*z)
    ifelse(is.finite(h), -h*exp(h), 0)
  }, -Inf, Inf, rel_tol, subdivisions)
  entropy <- log(base$scale) + log(base$integral$value) + zeta +
    entropy_cost$value / base$integral$value
  list(log_normalizer = base$log_value, u = u, phi = phi, zeta = zeta,
       entropy = entropy, A = A, B = B, d = d,
       identity_residual = B * phi - A * u - (d - 1),
       estimated_relative_integration_error = max(
         base$integral$abs_error / base$integral$value,
         if (!is.null(first$integral)) first$integral$abs_error / first$integral$value else NA_real_,
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
  compute <- function(use_score) .bnn_quad_tail_moments(K,C,mu,rho,rel_tol,subdivisions,use_score)
  tryCatch(compute(FALSE), gltbnn_quadrature_error = function(e) {
    tryCatch(compute(TRUE), gltbnn_quadrature_error = function(last) {
      stop("Could not resolve GLT tail quadrature for a layer with ", K,
           " scale groups in double precision (",conditionMessage(last),").",call.=FALSE)
    })
  })
}

.bnn_quad_tail_moments <- function(K, C, mu, rho, rel_tol, subdivisions, use_score) {
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
  boundary_case <- mode == boundary
  if (boundary_case && derivative(boundary) > 0) {
    # A narrow maximum can lie between the truncation boundary and the very
    # first grid point. For the fitted N(0,1) log-shape prior the log kernel,
    # written as a function of r=1/xi, is strictly concave on 0 < r <= 2.
    # Find the first descending zero even if both sampled heights miss it.
    right <- grid[which(derivative(grid) <= 0)[1L]]
    mode <- stats::uniroot(derivative, c(boundary, right), tol = 1e-14)$root
    peak <- log_kernel(mode)
  }
  rmode <- exp(-mode)
  gp <- derivative(mode) + (mode - mu) / rho^2
  curvature <- 1 / rho^2 + gp + K - K * rmode +
    K * rmode^2 * trigamma(1 + rmode) + K^2 * rmode^2 * trigamma(1 + K * rmode)
  if (mode == boundary) {
    scale <- 1 / (abs(derivative(boundary)) + sqrt(max(curvature, 1/rho^2)))
  } else {
    scale <- if (is.finite(curvature) && curvature > 0) 1 / sqrt(curvature) else rho
  }
  .bnn_quad_positive(scale, "Tail quadrature scale")
  lower_z <- (boundary - mode) / scale
  other_modes <- (candidates - mode) / scale
  breaks <- as.vector(outer(other_modes, c(-1, 0, 1), "+"))
  if (boundary_case) {
    # Do not insert the distant grid endpoint as a quadrature breakpoint.
    # An adaptive rule on [8, many thousands] can miss the exponential
    # shoulder of a boundary maximum and falsely report a tiny error.
    breaks <- c(-32,-16,-8,-4,-2,-1,0,1,2,4,8,16,32,64)
  }
  offset <- function(eta) {
    value <- log_kernel(eta) - peak
    if (use_score) {
      # If large nearly equal log densities defeat both QUADPACK tolerances,
      # integrate the score over the displacement from the mode. Four
      # Gauss–Legendre nodes resolve the smoothly varying score over each
      # short displacement. Distant tails can use direct subtraction.
      delta <- eta - mode
      near <- abs(delta) < .15
      h <- delta[near]
      nodes <- c(.0694318442029737,.3300094782075719,.6699905217924281,.9305681557970262)
      weights <- c(.1739274225687269,.3260725774312731,.3260725774312731,.1739274225687269)
      average <- numeric(length(h))
      for (j in seq_along(nodes))
        average <- average + weights[j]*derivative(mode + h*nodes[j])
      value[near] <- h*average
    }
    value
  }
  density <- function(z) {
    exponent <- offset(mode + scale * z)
    exponent[!is.finite(exponent)] <- -Inf
    exp(exponent)
  }
  # Large layers can lose a few digits subtracting the log densities at eta
  # and at the mode. Retry only a panel whose requested tolerance failed.
  roundoff_tol <- max(rel_tol, 1e-8)
  integral <- .bnn_quad_integrate(density, lower_z, Inf,
    rel_tol, subdivisions, extra = breaks, roundoff_tol = roundoff_tol)
  .bnn_quad_positive(integral$value, "Tail quadrature integral")
  retried <- integral$retried
  expectation <- function(f) {
    ans <- .bnn_quad_integrate(function(z) {
      eta <- mode + scale * z
      w <- density(z)
      result <- numeric(length(z))
      positive <- w > 0
      result[positive] <- f(eta[positive]) * w[positive]
      result
    }, lower_z, Inf, rel_tol, subdivisions, extra = breaks, roundoff_tol = roundoff_tol)
    if (ans$retried) retried <<- TRUE
    ans$value / integral$value
  }
  logZ <- peak + log(scale) + log(integral$value)
  log_support <- stats::pnorm((mu - boundary) / rho, log.p = TRUE)
  r <- if (boundary_case) {
    2 * (1 + expectation(function(eta) expm1(boundary - eta)))
  } else {
    expectation(function(eta) exp(-eta))
  }
  normal_log <- expectation(function(eta) stats::dnorm(eta, mu, rho, log = TRUE))
  # A large peak makes the individual log density and entropy contributions
  # poorly conditioned even if the requested integration tolerance succeeded.
  centered_entropy <- use_score || boundary_case || abs(peak)*rel_tol > 1e-4
  if (!centered_entropy && !retried)
    g_mean <- expectation(function(eta) .bnn_quad_tail_g(eta, K, C))
  if (centered_entropy || retried) {
    expected_offset <- expectation(offset)
    g_mean <- peak + expected_offset - normal_log
    entropy <- log(scale) + log(integral$value) - expected_offset
  } else {
    entropy <- logZ - normal_log - g_mean
  }
  if (!is.finite(r) || r <= 0 || r > 2) stop("Tail quadrature violated 0 < E[1/xi] <= 2.")
  list(log_normalizer = logZ, r = r, eta_mean = expectation(identity),
       log_prior_support = log_support, expected_log_reference = normal_log,
       expected_log_factor = g_mean, entropy_eta = entropy,
       C = C, K = K, mu = mu, rho = rho, mode = mode,
       estimated_relative_integration_error = integral$abs_error / integral$value)
}
