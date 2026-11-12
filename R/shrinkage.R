gltbnn_shrinkage <- function(fit) {
  if (!inherits(fit, "gltbnn_fit")) {
    stop("fit must be a model returned by gltbnn_fit().", call. = FALSE)
  }
  state <- fit$engine$state
  log_reference_variance <- if (is.null(state$noise)) 0 else state$noise$zeta
  coefficient_geometry <- identical(fit$engine$config$geometry, "coefficient")
  by_layer <- lapply(seq_along(fit$hidden), function(layer) {
    width <- fit$hidden[layer]
    dimension <- fit$engine$config$d[layer]
    group <- state$g[[layer]]
    if (coefficient_geometry) {
      unit <- rep(seq_len(width), each = dimension)
      coordinate <- rep(seq_len(dimension), width)
      # Scale factors follow the weight matrix's column-major storage order.
      index <- unit + (coordinate - 1L) * width
      rows <- data.frame(layer = layer, unit = unit, coordinate = coordinate,
                         is_bias = coordinate == dimension, group_size = 1L)
    } else {
      index <- seq_len(width)
      rows <- data.frame(layer = layer, unit = index, group_size = dimension)
    }
    terms <- .bnn_prior_terms(group, length(index), fit$prior)
    # The inverse-gamma factor represents tau under GLT and tau^2 under G/GL.
    # For GLT, tail$r = E_q[1/xi] and tail$eta_mean = E_q[log xi].
    log_layer_scale <- if (fit$prior == "GLT") group$tau$zeta else group$tau$zeta / 2
    rows$scale_precision <- terms$precision[index]
    rows$geometric_prior_sd <- exp((log_reference_variance + terms$logvar[index]) / 2)
    rows$geometric_layer_scale <- exp(log_layer_scale)
    rows$geometric_tail_shape <- if (fit$prior == "GLT") exp(group$tail$eta_mean) else NA_real_
    rows
  })
  do.call(rbind, by_layer)
}
