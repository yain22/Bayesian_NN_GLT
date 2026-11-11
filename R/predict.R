predict.gltbnn_fit <- function(object, newdata,
                              type = c("mean", "mean_draws", "response_draws",
                                       "mean_interval", "prediction_interval"),
                              draws = 256L, level = 0.95, seed = 1L, ...) {
  if (...length()) stop("Unused arguments supplied to predict().", call. = FALSE)
  type <- match.arg(type)
  draws <- .gltbnn_integer(draws, "draws")
  seed <- .gltbnn_integer(seed, "seed", minimum = 0L)
  interval <- type %in% c("mean_interval", "prediction_interval")
  if (interval) {
    if (draws < 2L) stop("Intervals require at least two draws.", call. = FALSE)
    if (!is.numeric(level) || length(level) != 1L || !is.null(dim(level)) ||
        !is.finite(level) || level <= 0 || level >= 1) {
      stop("level must be a number strictly between zero and one.", call. = FALSE)
    }
  }
  x <- .gltbnn_matrix(newdata, "newdata")
  if (ncol(x) != object$p) {
    stop("The number of predictors in newdata must match the training data.", call. = FALSE)
  }
  if (!is.null(object$predictor_names) && !is.null(colnames(x))) {
    if (!setequal(colnames(x), object$predictor_names)) {
      stop("Predictor names in newdata must match the training data.", call. = FALSE)
    }
    x <- x[, object$predictor_names, drop = FALSE]
  }
  standardized_x <- .gltbnn_scale_x(x, object$preprocess)
  predictive <- type %in% c("response_draws", "prediction_interval")
  samples <- .gltbnn_with_seed(seed, .gltbnn_engine_draws(
    object$engine, standardized_x, draws, predictive
  ))
  samples <- samples * object$preprocess$y_scale + object$preprocess$y_center
  rownames(samples) <- rownames(x)
  if (type == "mean") return(rowMeans(samples))
  if (interval) {
    tail_probability <- (1 - level) / 2
    bounds <- t(apply(samples, 1L, stats::quantile,
                      probs = c(tail_probability, 1 - tail_probability), names = FALSE))
    colnames(bounds) <- c("lower", "upper")
    return(bounds)
  }
  samples
}
