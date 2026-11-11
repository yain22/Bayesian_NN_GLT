gltbnn_shrinkage <- function(fit) {
  if (!inherits(fit, "gltbnn_fit")) {
    stop("fit must be a model returned by gltbnn_fit().", call. = FALSE)
  }
  state <- fit$engine$state
  by_layer <- lapply(seq_along(fit$hidden), function(layer) {
    width <- fit$hidden[layer]
    group <- state$g[[layer]]
    terms <- .bnn_prior_terms(group, width, fit$prior)
    # The inverse-gamma factor represents tau under GLT and tau^2 under G/GL.
    # For GLT, tail$r = E_q[1/xi] and tail$eta_mean = E_q[log xi].
    log_layer_scale <- if (fit$prior == "GLT") group$tau$zeta else group$tau$zeta / 2
    data.frame(layer = layer, unit = seq_len(width),
               group_size = fit$engine$config$d[layer],
               scale_precision = terms$precision,
               geometric_prior_sd = exp((state$noise$zeta + terms$logvar) / 2),
               geometric_layer_scale = exp(log_layer_scale),
               geometric_tail_shape = if (fit$prior == "GLT") exp(group$tail$eta_mean) else NA_real_)
  })
  do.call(rbind, by_layer)
}
