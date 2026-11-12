# Rows of each weight matrix correspond to receiving units. The last column
# multiplies the constant input 1, so a bias belongs to its unit's group.
.bnn_forward <- function(x, weights) {
  features <- pre <- vector("list", length(weights))
  h <- x
  for (layer in seq_along(weights)) {
    features[[layer]] <- cbind(h, 1)
    pre[[layer]] <- features[[layer]] %*% t(weights[[layer]])
    h <- if (layer < length(weights)) pmax(pre[[layer]], 0) else pre[[layer]]
  }
  list(y = if (ncol(h) == 1L) as.vector(h) else h, logits = h, features = features, pre = pre)
}

.bnn_backprop <- function(back, forward, weights) {
  gradient <- vector("list", length(weights))
  for (layer in rev(seq_along(weights))) {
    gradient[[layer]] <- crossprod(back, forward$features[[layer]])
    if (layer > 1L) {
      w <- weights[[layer]]
      back <- (back %*% w[, -ncol(w), drop = FALSE]) * (forward$pre[[layer - 1L]] > 0)
    }
  }
  gradient
}

# Full-batch likelihood warm start shared by all three shrinkage priors.
.bnn_initialize <- function(x, y, hidden, seed, steps, outcome = "continuous", n_classes = NULL) {
  set.seed(seed)
  inputs <- c(ncol(x), hidden) + 1L
  output_units <- if (outcome == "categorical") n_classes - 1L else 1L
  units <- c(hidden, output_units)
  weights <- lapply(seq_along(units), function(l) {
    matrix(stats::rnorm(units[l] * inputs[l], sd = sqrt(2 / inputs[l])), nrow = units[l])
  })
  first <- second <- lapply(weights, function(w) w * 0)
  for (iteration in seq_len(steps)) {
    forward <- .bnn_forward(x, weights)
    if (outcome == "continuous") {
      back <- matrix((forward$y - y) / length(y), ncol = 1L)
    } else {
      log_probs <- .bnn_class_log_probs(forward$logits, outcome)
      back <- -.bnn_class_score(log_probs, y, outcome) / length(y)
    }
    gradient <- .bnn_backprop(back, forward, weights)
    for (l in seq_along(weights)) {
      first[[l]] <- .9 * first[[l]] + .1 * gradient[[l]]
      second[[l]] <- .999 * second[[l]] + .001 * gradient[[l]]^2
      weights[[l]] <- weights[[l]] - .003 * (first[[l]] / (1 - .9^iteration)) /
        (sqrt(second[[l]] / (1 - .999^iteration)) + 1e-8)
    }
  }
  weights
}

.bnn_draw_weights <- function(parameters, epsilon) {
  Map(function(m, s, z) m + exp(s) * z, parameters$mu, parameters$logsd, epsilon)
}

.bnn_normal_draw <- function(parameters) {
  lapply(parameters$mu, function(w) matrix(stats::rnorm(length(w)), nrow(w), ncol(w)))
}

.bnn_antithetic_draws <- function(parameters, draws) {
  positive <- lapply(seq_len(draws / 2L), function(i) .bnn_normal_draw(parameters))
  c(positive, lapply(positive, function(draw) lapply(draw, function(z) -z)))
}

# Row sums give E[||w_kl||^2] from the diagonal variational covariance.
.bnn_weight_moments <- function(parameters, coefficient = FALSE) {
  Map(function(m, s) {
    moments <- m^2 + exp(2 * s)
    if (coefficient) moments else rowSums(moments)
  }, parameters$mu, parameters$logsd)
}

.bnn_sse <- function(parameters, x, y, epsilon) {
  vapply(epsilon, function(z) {
    prediction <- .bnn_forward(x, .bnn_draw_weights(parameters, z))$y
    sum((y - prediction)^2)
  }, numeric(1))
}

.gltbnn_engine_draws <- function(fit, x, draws, predictive) {
  result <- matrix(0, nrow = nrow(x), ncol = draws)
  for (b in seq_len(draws)) {
    weights <- .bnn_draw_weights(fit$parameters, .bnn_normal_draw(fit$parameters))
    result[, b] <- .bnn_forward(x, weights)$y
  }
  if (predictive) {
    # One posterior noise variance per joint draw, with independent errors by row.
    precision <- stats::rgamma(draws, shape = fit$state$noise$a, rate = fit$state$noise$b)
    errors <- matrix(stats::rnorm(length(result)), nrow = nrow(x), ncol = draws)
    result <- result + sweep(errors, 2L, sqrt(precision), "/")
  }
  result
}
