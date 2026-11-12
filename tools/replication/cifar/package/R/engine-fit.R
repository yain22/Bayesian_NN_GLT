# Fixed-scale weight objective and its reparameterization gradient.
.bnn_weight_objective <- function(parameters, x, y, phi, rates, epsilon, gradient = FALSE,
                                  outcome = "continuous") {
  moments <- .bnn_weight_moments(parameters, coefficient = TRUE)
  value <- sum(vapply(parameters$logsd, sum, numeric(1))) -
    .5 * sum(vapply(seq_along(moments), function(l) sum(rates[[l]] * moments[[l]]), numeric(1)))
  if (gradient) gm <- gs <- lapply(parameters$mu, function(m) m * 0)
  draws <- length(epsilon)
  for (z in epsilon) {
    weights <- .bnn_draw_weights(parameters, z)
    forward <- .bnn_forward(x, weights)
    if (outcome == "continuous") {
      residual <- y - forward$y
      value <- value - .5 * phi * sum(residual^2) / draws
      if (gradient) score <- matrix(phi * residual, ncol = 1L)
    } else {
      log_probs <- .bnn_class_log_probs(forward$logits, outcome)
      value <- value + .bnn_class_loglik(log_probs, y) / draws
      if (gradient) score <- .bnn_class_score(log_probs, y, outcome)
    }
    if (gradient) {
      back <- .bnn_backprop(score, forward, weights)
      for (l in seq_along(weights)) {
        grad <- back[[l]] / draws
        gm[[l]] <- gm[[l]] + grad
        gs[[l]] <- gs[[l]] + grad * exp(parameters$logsd[[l]]) * z[[l]]
      }
    }
  }
  if (!gradient) return(value)
  for (l in seq_along(gm)) {
    gm[[l]] <- gm[[l]] - rates[[l]] * parameters$mu[[l]]
    gs[[l]] <- gs[[l]] + 1 - rates[[l]] * exp(2 * parameters$logsd[[l]])
  }
  list(value = value, gradient = c(gm, gs))
}

.bnn_pair_means <- function(x) {
  h <- length(x) / 2L
  (x[seq_len(h)] + x[h + seq_len(h)]) / 2
}

# Full augmented-prior ELBO with antithetic draws for the likelihood term.
.bnn_elbo <- function(state, parameters, x, y, priors, epsilon, noise_prior = NULL,
                      outcome = "continuous") {
  moments <- .bnn_weight_moments(parameters, coefficient = TRUE)
  output <- length(moments)
  gaussian_entropy <- function(ls) sum((1 + log(2 * pi)) / 2 + ls)
  output_dim <- length(parameters$mu[[output]])
  hidden_log_scale_sq <- 0
  hidden_precision <- 1
  noise_terms <- NULL
  if (outcome == "continuous") {
    noise <- state$noise
    hidden_log_scale_sq <- noise$zeta
    hidden_precision <- noise$phi
    sse <- .bnn_sse(parameters, x, y, epsilon)
    observation <- -length(y) * (log(2 * pi) + noise$zeta) / 2 - noise$phi * mean(sse) / 2
    centered_draws <- noise$phi * (mean(sse) - sse) / 2
    noise_terms <- if (is.null(noise_prior)) {
      c(noise_prior = -noise$zeta, noise_entropy = noise$entropy)
    } else {
      c(noise_neg_kl = -.bnn_ig_kl(noise, noise_prior[["shape"]], noise_prior[["scale"]]))
    }
  } else {
    loglik <- vapply(epsilon, function(z) {
      weights <- .bnn_draw_weights(parameters, z)
      logits <- .bnn_forward(x, weights)$logits
      .bnn_class_loglik(.bnn_class_log_probs(logits, outcome), y)
    }, numeric(1))
    observation <- mean(loglik)
    centered_draws <- loglik - observation
  }
  terms <- c(
    observation = observation,
    noise_terms,
    output_weight_prior = -output_dim * (log(2 * pi) + state$u$zeta) / 2 -
      state$u$phi * sum(moments[[output]]) / 2,
    output_weight_entropy = gaussian_entropy(parameters$logsd[[output]]),
    output_u_prior = .bnn_ig_log_prior(state$u, .5, state$rho$phi, -state$rho$zeta),
    output_rho_prior = .bnn_ig_log_prior(state$rho, .5, 1 / 25, log(1 / 25)),
    output_hyper_entropy = state$u$entropy + state$rho$entropy,
    hidden_weight_prior = 0, hidden_entropy = 0, prior_hyper = 0
  )
  for (l in seq_along(priors)) {
    pr <- priors[[l]]
    terms["hidden_weight_prior"] <- terms["hidden_weight_prior"] - .5 * sum(
      log(2 * pi) + hidden_log_scale_sq + pr$logvar + hidden_precision * moments[[l]] * pr$precision)
    terms["hidden_entropy"] <- terms["hidden_entropy"] + gaussian_entropy(parameters$logsd[[l]])
    terms["prior_hyper"] <- terms["prior_hyper"] + pr$hyper
  }
  if (any(!is.finite(terms))) stop("Nonfinite ELBO component during fitting.")
  value <- sum(terms)
  pairs <- .bnn_pair_means(centered_draws)
  summary <- if (outcome == "continuous") list(expected_sse = mean(sse)) else list(expected_loglik = observation)
  c(list(value = value, components = terms), summary,
    list(mcse = stats::sd(pairs) / sqrt(length(pairs)), sample_values = value + centered_draws))
}

.bnn_adam_step <- function(parameters, adam, iteration, x, y, hidden_precision, rates, control,
                           outcome = "continuous", n_total = nrow(x)) {
  epsilon <- .bnn_antithetic_draws(parameters, control$weight_draws)
  likelihood_precision <- hidden_precision * (n_total / nrow(x))
  objective <- .bnn_weight_objective(parameters, x, y, likelihood_precision, rates, epsilon, TRUE, outcome)
  grad <- lapply(objective$gradient, function(g) g / n_total)
  norm <- sqrt(sum(vapply(grad, function(g) sum(g^2), numeric(1))))
  if (!is.finite(objective$value)) stop("Nonfinite weight objective.")
  if (!is.finite(norm)) {
    # Finite gradient entries can overflow when squared. The unit direction
    # and the clipping decision both remain available on a rescaled vector.
    largest <- max(vapply(grad, function(g) max(abs(g)), numeric(1)))
    if (!is.finite(largest)) stop("Nonfinite weight gradient.")
    scaled <- lapply(grad, function(g) g / largest)
    scaled_norm <- sqrt(sum(vapply(scaled, function(g) sum(g^2), numeric(1))))
    norm <- largest * scaled_norm
    if (norm > control$grad_clip) {
      grad <- lapply(scaled, function(g) (g / scaled_norm) * control$grad_clip)
    }
  } else if (norm > control$grad_clip) {
    grad <- lapply(grad, function(g) g * control$grad_clip / norm)
  }
  adam$m <- Map(function(m, g) .9 * m + .1 * g, adam$m, grad)
  adam$v <- Map(function(v, g) .999 * v + .001 * g^2, adam$v, grad)
  denominator <- lapply(adam$v, function(v) sqrt(v / (1 - .999^iteration)) + 1e-8)
  direction <- Map(function(m, den) (m / (1 - .9^iteration)) / den, adam$m, denominator)
  slopes <- Map(function(g, d) sum(g * d), grad, direction)
  fallback <- sum(unlist(slopes)) <= 0
  if (fallback) direction <- Map(`/`, grad, denominator)

  # Reuse the gradient draws for candidate scoring so that sampling noise
  # cannot produce spurious improvements within this backtracking step.
  accepted <- FALSE
  gain <- 0
  indices <- seq_along(parameters$mu)
  for (bt in 0:control$max_backtracks) {
    scale <- control$lr * 2^(-bt)
    candidate <- Map(function(p, d) p + scale * d, c(parameters$mu, parameters$logsd), direction)
    cp <- list(mu = candidate[indices], logsd = candidate[length(indices) + indices])
    cp$logsd <- lapply(cp$logsd, function(z) {
      z[] <- pmax(control$logsd_bounds[1], pmin(control$logsd_bounds[2], as.vector(z)))
      z
    })
    value <- .bnn_weight_objective(cp, x, y, likelihood_precision, rates, epsilon, outcome = outcome)
    if (is.finite(value) && value >= objective$value) {
      parameters <- cp
      gain <- value - objective$value
      accepted <- TRUE
      break
    }
  }
  list(parameters = parameters, adam = adam,
       record = data.frame(step = iteration, accepted = accepted, backtracks = bt,
                           direction_fallback = fallback, sample_objective = objective$value,
                           sample_gain = gain, gradient_norm = norm))
}

.gltbnn_engine_fit <- function(x, y, hidden, prior, control, seed, noise_prior = NULL,
                               outcome = "continuous", n_classes = NULL, geometry = "coefficient",
                               initial_weights = NULL, progress = NULL, fixed_shape = NULL,
                               batches = NULL, milestones = integer(), on_milestone = NULL) {
  started <- proc.time()[["elapsed"]]
  continuous <- outcome == "continuous"
  batch_fit <- !is.null(batches)
  if (length(milestones) && (!is.function(on_milestone) || !is.numeric(milestones) ||
      any(!is.finite(milestones) | milestones != floor(milestones) |
          milestones < 1L | milestones > control$outer))) stop("Invalid milestone cycles or callback.")
  if (batch_fit) {
    if (!continuous || !is.list(batches) || length(batches) != control$outer * control$weight_steps ||
        any(!vapply(batches, function(ids) is.numeric(ids) && length(ids) > 0L &&
          all(is.finite(ids) & ids == floor(ids) & ids >= 1L & ids <= nrow(x)), logical(1)))) {
      stop("A Gaussian minibatch fit requires one nonempty vector of training indices per weight step.")
    }
  }
  shape0 <- if (is.null(noise_prior)) 0 else noise_prior[["shape"]]
  scale0 <- if (is.null(noise_prior)) 0 else noise_prior[["scale"]]
  dimensions <- c(ncol(x), hidden[-length(hidden)]) + 1L
  coefficient <- geometry == "coefficient"
  if (!is.null(fixed_shape) && (!coefficient || prior != "GLT" || length(fixed_shape) != 1L ||
      !is.finite(fixed_shape) || fixed_shape <= .5)) stop("A fixed tail requires coefficient GLT and shape > 1/2.")
  group_counts <- if (coefficient) as.double(hidden) * dimensions else hidden
  group_sizes <- if (coefficient) rep(1L, length(hidden)) else dimensions
  weights <- initial_weights
  if (is.null(weights)) weights <- .bnn_initialize(x, y, hidden, seed, control$init_steps, outcome, n_classes)
  output <- length(weights)
  initial_logsd <- max(control$logsd_bounds[1L], min(log(.01), control$logsd_bounds[2L]))
  parameters <- list(mu = weights, logsd = lapply(weights, function(w) matrix(initial_logsd, nrow(w), ncol(w))))
  state <- list(g = Map(function(k, d) .bnn_init_groups(k, d, prior, geometry, fixed_shape), group_counts, group_sizes),
                u = .bnn_ig(2, 2), rho = .bnn_ig(1, 1.04))
  hidden_precision <- 1
  if (continuous) {
    state$noise <- .bnn_ig(3, 1)
    hidden_precision <- state$noise$phi
  }
  layer_priors <- function(groups) Map(function(g, k) .bnn_prior_terms(g, k, prior), groups, group_counts)
  priors <- layer_priors(state$g)
  set.seed(as.integer(as.double(seed) %% 2147483646 + 1))
  # For scheduled large-data experiments, the archived protocol monitors the
  # pooled batch SSE rather than materializing full-data ELBO draws.
  if (batch_fit) {
    monitor <- list(value = NA_real_, mcse = NA_real_, components = numeric())
  } else {
    monitor_draws <- .bnn_antithetic_draws(parameters, control$monitor_draws)
    monitor <- .bnn_elbo(state, parameters, x, y, priors, monitor_draws, noise_prior, outcome)
  }
  trace <- list(data.frame(outer = 0L, elbo = monitor$value, elbo_mcse = monitor$mcse,
                           gain = NA_real_, paired_gain_mcse = NA_real_, noise_precision = hidden_precision,
                           accepted = 0L, rejected = 0L, backtracks = 0L, direction_fallbacks = 0L,
                           seconds = proc.time()[["elapsed"]] - started))
  components <- if (batch_fit) list() else list(data.frame(outer = 0L, as.list(monitor$components), check.names = FALSE))
  if (batch_fit) trace[[1L]]$pooled_batch_rmse <- NA_real_
  steps <- vector("list", control$outer)
  finish <- function(done) {
    set.seed(as.integer((as.double(seed) + 100000) %% 2147483646 + 1))
    endpoint <- if (batch_fit) {
      list(value = NA_real_, mcse = NA_real_, note = "The pooled minibatch protocol omits the full-data ELBO.")
    } else {
      .bnn_elbo(state, parameters, x, y, priors,
                .bnn_antithetic_draws(parameters, control$endpoint_draws), noise_prior, outcome)
    }
    endpoint$sample_values <- NULL
    history <- do.call(rbind, trace)
    if (!continuous) history$noise_precision <- NULL
    ctl <- control
    ctl$outer <- done
    list(state = state, parameters = parameters, config = list(n = nrow(x), p = ncol(x), K = hidden,
         L = length(hidden), d = dimensions, M = as.double(hidden) * dimensions, geometry = geometry,
         prior = prior, outcome = outcome, n_classes = n_classes, fixed_shape = fixed_shape), control = ctl,
         trace = history, components = do.call(rbind, components),
         weight_steps = do.call(rbind, steps[seq_len(done)]), endpoint = endpoint,
         seed = seed, status = "fixed_budget", outer = done, seconds = proc.time()[["elapsed"]] - started)
  }
  zero <- lapply(c(parameters$mu, parameters$logsd), function(z) z * 0)
  adam <- list(m = zero, v = zero)
  iteration <- 0L
  for (outer in seq_len(control$outer)) {
    rates <- c(lapply(priors, function(pr) hidden_precision * pr$precision), list(state$u$phi))
    block <- vector("list", control$weight_steps)
    for (j in seq_len(control$weight_steps)) {
      iteration <- iteration + 1L
      if (batch_fit) {
        ids <- batches[[iteration]]
        update <- .bnn_adam_step(parameters, adam, iteration, x[ids, , drop = FALSE], y[ids],
                                 hidden_precision, rates, control, outcome, n_total = nrow(x))
      } else {
        update <- .bnn_adam_step(parameters, adam, iteration, x, y, hidden_precision, rates, control, outcome)
      }
      parameters <- update$parameters
      adam <- update$adam
      block[[j]] <- data.frame(outer = outer, update$record)
    }
    steps[[outer]] <- do.call(rbind, block)
    moments <- .bnn_weight_moments(parameters, coefficient)
    if (continuous) {
      residual_draws <- .bnn_antithetic_draws(parameters, control$noise_draws)
      if (batch_fit) {
        cycle_steps <- (outer - 1L) * control$weight_steps + seq_len(control$weight_steps)
        seen <- unlist(batches[cycle_steps], use.names = FALSE)
        sse <- mean(.bnn_sse(parameters, x[seen, , drop = FALSE], y[seen], residual_draws)) *
          (nrow(x) / length(seen))
      } else {
        sse <- mean(.bnn_sse(parameters, x, y, residual_draws))
      }
      penalty <- sum(vapply(seq_along(hidden), function(l) sum(moments[[l]] * priors[[l]]$precision), numeric(1)))
      state$noise <- .bnn_ig(shape0 + (length(y) + sum(hidden * dimensions)) / 2,
                             scale0 + (sse + penalty) / 2)
      hidden_precision <- state$noise$phi
    }
    state$u <- .bnn_ig((length(weights[[output]]) + 1) / 2, sum(moments[[output]]) / 2 + state$rho$phi)
    state$rho <- .bnn_ig(1, state$u$phi + 1 / 25)
    for (l in seq_along(hidden)) {
      state$g[[l]] <- .bnn_update_groups(state$g[[l]], group_counts[l], group_sizes[l], prior,
                                         moments[[l]], hidden_precision)
    }
    priors <- layer_priors(state$g)
    if (batch_fit) {
      next_monitor <- monitor
      gain_mcse <- NA_real_
    } else {
      next_monitor <- .bnn_elbo(state, parameters, x, y, priors, monitor_draws, noise_prior, outcome)
      paired_gain <- .bnn_pair_means(next_monitor$sample_values - monitor$sample_values)
      gain_mcse <- stats::sd(paired_gain) / sqrt(length(paired_gain))
    }
    records <- steps[[outer]]
    trace[[outer + 1L]] <- data.frame(
      outer = outer, elbo = next_monitor$value, elbo_mcse = next_monitor$mcse,
      gain = next_monitor$value - monitor$value, paired_gain_mcse = gain_mcse,
      noise_precision = hidden_precision, accepted = sum(records$accepted), rejected = sum(!records$accepted),
      backtracks = sum(records$backtracks), direction_fallbacks = sum(records$direction_fallback),
      seconds = proc.time()[["elapsed"]] - started)
    if (batch_fit) trace[[outer + 1L]]$pooled_batch_rmse <- sqrt(sse / nrow(x))
    if (!batch_fit) {
      components[[outer + 1L]] <- data.frame(outer = outer, as.list(next_monitor$components), check.names = FALSE)
    }
    monitor <- next_monitor
    if (!is.null(progress)) progress(trace[[outer + 1L]])
    if (outer %in% milestones) {
      # A callback may itself draw predictions; protect the optimizer stream.
      .gltbnn_with_seed(seed, on_milestone(outer, finish(outer)))
    }
  }
  finish(control$outer)
}
