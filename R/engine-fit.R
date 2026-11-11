# Fixed-scale weight objective and its reparameterization gradient.
.bnn_weight_objective <- function(parameters, x, y, phi, rates, epsilon, gradient = FALSE) {
  moments <- .bnn_weight_moments(parameters)
  value <- sum(vapply(parameters$logsd, sum, numeric(1))) -
    .5 * sum(vapply(seq_along(moments), function(l) sum(rates[[l]] * moments[[l]]), numeric(1)))
  if (gradient) gm <- gs <- lapply(parameters$mu, function(m) m * 0)
  draws <- length(epsilon)
  for (z in epsilon) {
    weights <- .bnn_draw_weights(parameters, z)
    forward <- .bnn_forward(x, weights)
    residual <- y - forward$y
    value <- value - .5 * phi * sum(residual^2) / draws
    if (gradient) {
      back <- .bnn_backprop(matrix(phi * residual, ncol = 1L), forward, weights)
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

# Full augmented-prior ELBO. Expected squared error is the Monte Carlo term.
.bnn_elbo <- function(state, parameters, x, y, priors, epsilon, noise_prior = NULL) {
  noise <- state$noise
  sse <- .bnn_sse(parameters, x, y, epsilon)
  moments <- .bnn_weight_moments(parameters)
  output <- length(moments)
  gaussian_entropy <- function(ls) sum((1 + log(2 * pi)) / 2 + ls)
  output_dim <- length(parameters$mu[[output]])
  terms <- c(
    observation = -length(y) * (log(2 * pi) + noise$zeta) / 2 - noise$phi * mean(sse) / 2,
    noise_prior = if (is.null(noise_prior)) -noise$zeta else
      .bnn_ig_log_prior(noise, noise_prior[["shape"]], noise_prior[["scale"]], log(noise_prior[["scale"]])),
    noise_entropy = noise$entropy,
    output_weight_prior = -output_dim * (log(2 * pi) + state$u$zeta) / 2 -
      state$u$phi * sum(moments[[output]]) / 2,
    output_weight_entropy = gaussian_entropy(parameters$logsd[[output]]),
    output_u_prior = .bnn_ig_log_prior(state$u, .5, state$rho$phi, -state$rho$zeta),
    output_rho_prior = .bnn_ig_log_prior(state$rho, .5, 1 / 25, log(1 / 25)),
    output_hyper_entropy = state$u$entropy + state$rho$entropy,
    hidden_weight_prior = 0, hidden_entropy = 0, prior_hyper = 0
  )
  for (l in seq_along(priors)) {
    d <- ncol(parameters$mu[[l]])
    pr <- priors[[l]]
    terms["hidden_weight_prior"] <- terms["hidden_weight_prior"] - .5 * sum(
      d * (log(2 * pi) + noise$zeta + pr$logvar) + noise$phi * moments[[l]] * pr$precision)
    terms["hidden_entropy"] <- terms["hidden_entropy"] + gaussian_entropy(parameters$logsd[[l]])
    terms["prior_hyper"] <- terms["prior_hyper"] + pr$hyper
  }
  if (any(!is.finite(terms))) stop("Nonfinite ELBO component during fitting.")
  value <- sum(terms)
  pair_sse <- .bnn_pair_means(sse)
  list(value = value, components = terms, expected_sse = mean(sse),
       mcse = noise$phi * stats::sd(pair_sse) / (2 * sqrt(length(pair_sse))),
       sample_values = value + noise$phi * (mean(sse) - sse) / 2)
}

.bnn_adam_step <- function(parameters, adam, iteration, x, y, noise_phi, rates, control) {
  epsilon <- .bnn_antithetic_draws(parameters, control$weight_draws)
  objective <- .bnn_weight_objective(parameters, x, y, noise_phi, rates, epsilon, TRUE)
  grad <- lapply(objective$gradient, function(g) g / nrow(x))
  norm <- sqrt(sum(vapply(grad, function(g) sum(g^2), numeric(1))))
  if (!is.finite(norm) || !is.finite(objective$value)) stop("Nonfinite weight objective or gradient.")
  if (norm > control$grad_clip) grad <- lapply(grad, function(g) g * control$grad_clip / norm)
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
    value <- .bnn_weight_objective(cp, x, y, noise_phi, rates, epsilon)
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

.gltbnn_engine_fit <- function(x, y, hidden, prior, control, seed, noise_prior = NULL) {
  started <- proc.time()[["elapsed"]]
  shape0 <- if (is.null(noise_prior)) 0 else noise_prior[["shape"]]
  scale0 <- if (is.null(noise_prior)) 0 else noise_prior[["scale"]]
  dimensions <- c(ncol(x), hidden[-length(hidden)]) + 1L
  weights <- .bnn_initialize(x, y, hidden, seed, control$init_steps)
  output <- length(weights)
  initial_logsd <- max(control$logsd_bounds[1L], min(log(.01), control$logsd_bounds[2L]))
  parameters <- list(mu = weights, logsd = lapply(weights, function(w) matrix(initial_logsd, nrow(w), ncol(w))))
  state <- list(g = Map(function(k, d) .bnn_init_groups(k, d, prior), hidden, dimensions),
                noise = .bnn_ig(3, 1), u = .bnn_ig(2, 2), rho = .bnn_ig(1, 1.04))
  layer_priors <- function(groups) Map(function(g, k) .bnn_prior_terms(g, k, prior), groups, hidden)
  priors <- layer_priors(state$g)
  set.seed(as.integer(as.double(seed) %% 2147483646 + 1))
  monitor_draws <- .bnn_antithetic_draws(parameters, control$monitor_draws)
  monitor <- .bnn_elbo(state, parameters, x, y, priors, monitor_draws, noise_prior)
  trace <- list(data.frame(outer = 0L, elbo = monitor$value, elbo_mcse = monitor$mcse,
                           gain = NA_real_, paired_gain_mcse = NA_real_, noise_precision = state$noise$phi,
                           accepted = 0L, rejected = 0L, backtracks = 0L, direction_fallbacks = 0L,
                           seconds = proc.time()[["elapsed"]] - started))
  components <- list(data.frame(outer = 0L, as.list(monitor$components), check.names = FALSE))
  steps <- vector("list", control$outer)
  zero <- lapply(c(parameters$mu, parameters$logsd), function(z) z * 0)
  adam <- list(m = zero, v = zero)
  iteration <- 0L
  for (outer in seq_len(control$outer)) {
    rates <- c(lapply(priors, function(pr) state$noise$phi * pr$precision), list(state$u$phi))
    block <- vector("list", control$weight_steps)
    for (j in seq_len(control$weight_steps)) {
      iteration <- iteration + 1L
      update <- .bnn_adam_step(parameters, adam, iteration, x, y, state$noise$phi, rates, control)
      parameters <- update$parameters
      adam <- update$adam
      block[[j]] <- data.frame(outer = outer, update$record)
    }
    steps[[outer]] <- do.call(rbind, block)
    moments <- .bnn_weight_moments(parameters)
    residual_draws <- .bnn_antithetic_draws(parameters, control$noise_draws)
    sse <- mean(.bnn_sse(parameters, x, y, residual_draws))
    penalty <- sum(vapply(seq_along(hidden), function(l) sum(moments[[l]] * priors[[l]]$precision), numeric(1)))
    state$noise <- .bnn_ig(shape0 + (length(y) + sum(hidden * dimensions)) / 2,
                           scale0 + (sse + penalty) / 2)
    state$u <- .bnn_ig((length(weights[[output]]) + 1) / 2, sum(moments[[output]]) / 2 + state$rho$phi)
    state$rho <- .bnn_ig(1, state$u$phi + 1 / 25)
    for (l in seq_along(hidden)) {
      state$g[[l]] <- .bnn_update_groups(state$g[[l]], hidden[l], dimensions[l], prior,
                                         moments[[l]], state$noise$phi)
    }
    priors <- layer_priors(state$g)
    next_monitor <- .bnn_elbo(state, parameters, x, y, priors, monitor_draws, noise_prior)
    paired_gain <- .bnn_pair_means(next_monitor$sample_values - monitor$sample_values)
    records <- steps[[outer]]
    trace[[outer + 1L]] <- data.frame(
      outer = outer, elbo = next_monitor$value, elbo_mcse = next_monitor$mcse,
      gain = next_monitor$value - monitor$value, paired_gain_mcse = stats::sd(paired_gain) / sqrt(length(paired_gain)),
      noise_precision = state$noise$phi, accepted = sum(records$accepted), rejected = sum(!records$accepted),
      backtracks = sum(records$backtracks), direction_fallbacks = sum(records$direction_fallback),
      seconds = proc.time()[["elapsed"]] - started)
    components[[outer + 1L]] <- data.frame(outer = outer, as.list(next_monitor$components), check.names = FALSE)
    monitor <- next_monitor
  }
  set.seed(as.integer((as.double(seed) + 100000) %% 2147483646 + 1))
  endpoint <- .bnn_elbo(state, parameters, x, y, priors,
                        .bnn_antithetic_draws(parameters, control$endpoint_draws), noise_prior)
  endpoint$sample_values <- NULL
  list(state = state, parameters = parameters, config = list(n = nrow(x), p = ncol(x), K = hidden,
       L = length(hidden), d = dimensions, prior = prior), control = control,
       trace = do.call(rbind, trace), components = do.call(rbind, components),
       weight_steps = do.call(rbind, steps), endpoint = endpoint,
       seed = seed, status = "fixed_budget", outer = control$outer, seconds = proc.time()[["elapsed"]] - started)
}
