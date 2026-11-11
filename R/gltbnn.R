# Fit a Gaussian-response neural network with grouped shrinkage priors.
gltbnn_fit <- function(x, y, hidden = c(10L, 10L), prior = c("GLT", "GL", "G"),
                       control = list(), seed = 1L, noise_prior = NULL) {
  prior <- match.arg(prior)
  x <- .gltbnn_matrix(x, "x")
  if (nrow(x) < 2L || !is.numeric(y) || !is.null(dim(y)) ||
      length(y) != nrow(x) || any(!is.finite(y))) {
    stop("Supply at least two observations and one finite numeric response per row of x.",
         call. = FALSE)
  }
  if (!is.numeric(hidden) || !is.null(dim(hidden)) || !length(hidden) ||
      any(!is.finite(hidden)) || any(hidden != floor(hidden)) ||
      any(hidden < 1 | hidden > .Machine$integer.max)) {
    stop("hidden must contain the positive integer width of each hidden layer.",
         call. = FALSE)
  }
  hidden <- as.integer(hidden)
  control <- .gltbnn_control(control)
  seed <- .gltbnn_integer(seed, "seed", minimum = 0L)
  if (!is.null(noise_prior)) {
    if (!is.numeric(noise_prior) || !is.null(dim(noise_prior)) ||
        length(noise_prior) != 2L || !setequal(names(noise_prior), c("shape", "scale")) ||
        any(!is.finite(noise_prior)) || any(noise_prior <= 0)) {
      stop("noise_prior must be NULL or a positive finite numeric vector named shape and scale.",
           call. = FALSE)
    }
    noise_prior <- noise_prior[c("shape", "scale")]
  }
  prep <- list(x_center = colMeans(x), x_scale = apply(x, 2L, stats::sd),
               y_center = mean(y), y_scale = stats::sd(y))
  if (any(!is.finite(prep$x_scale) | prep$x_scale <= 0) ||
      !is.finite(prep$y_scale) || prep$y_scale <= 0) {
    stop("Each training predictor and the response must have a positive finite sample standard deviation.",
         call. = FALSE)
  }

  standardized_x <- .gltbnn_scale_x(x, prep)
  standardized_y <- (as.numeric(y) - prep$y_center) / prep$y_scale
  engine <- .gltbnn_with_seed(seed, .gltbnn_engine_fit(
    standardized_x, standardized_y, hidden, prior, control, seed, noise_prior
  ))
  structure(list(prior = prior, hidden = hidden, n = nrow(x), p = ncol(x),
                 predictor_names = colnames(x), preprocess = prep, seed = seed,
                 control = control, noise_prior = noise_prior, engine = engine), class = "gltbnn_fit")
}

print.gltbnn_fit <- function(x, ...) {
  widths <- paste(x$hidden, collapse = ", ")
  cat(sprintf("Bayesian neural network for Gaussian regression (%s prior)\n", x$prior),
      sprintf("Training dimensions (observations x predictors): %d x %d\n", x$n, x$p),
      sprintf("Hidden layer widths: %s\n", widths),
      sprintf("Outer iterations: %d\n", x$control$outer), sep = "")
  invisible(x)
}

.gltbnn_matrix <- function(x, label) {
  if (is.data.frame(x)) {
    numeric_columns <- vapply(x, function(column) {
      is.numeric(column) && is.null(dim(column))
    }, logical(1))
    if (all(numeric_columns)) x <- as.matrix(x)
  }
  if (!is.matrix(x) || !is.numeric(x) || !nrow(x) || !ncol(x) || any(!is.finite(x))) {
    stop(label, " must be a nonempty numeric matrix or numeric data frame with finite entries.",
         call. = FALSE)
  }
  nms <- colnames(x)
  if (!is.null(nms) && (anyNA(nms) || any(!nzchar(nms)) || anyDuplicated(nms))) {
    stop(label, " must have unique, nonempty column names when names are supplied.",
         call. = FALSE)
  }
  storage.mode(x) <- "double"
  x
}

.gltbnn_scale_x <- function(x, prep) {
  sweep(sweep(x, 2L, prep$x_center, "-"), 2L, prep$x_scale, "/")
}

.gltbnn_integer <- function(value, name, minimum = 1L) {
  if (!is.numeric(value) || length(value) != 1L || !is.null(dim(value)) ||
      !is.finite(value) || value != floor(value) ||
      value < minimum || value > .Machine$integer.max) {
    stop(name, " must be an integer of at least ", minimum, " within R's integer range.",
         call. = FALSE)
  }
  as.integer(value)
}

.gltbnn_control <- function(control) {
  defaults <- list(outer = 60L, weight_steps = 10L, init_steps = 600L,
                   lr = 0.003, weight_draws = 4L, noise_draws = 16L,
                   monitor_draws = 64L, endpoint_draws = 256L, grad_clip = 10,
                   logsd_bounds = c(-15, 3), max_backtracks = 10L)
  if (!is.list(control) || is.data.frame(control) ||
      (length(control) && (is.null(names(control)) || anyDuplicated(names(control)) ||
                            !all(names(control) %in% names(defaults))))) {
    stop("control must be a list with unique names selected from the fitting controls.",
         call. = FALSE)
  }
  defaults[names(control)] <- control
  for (nm in c("outer", "weight_steps", "init_steps", "max_backtracks")) {
    defaults[[nm]] <- .gltbnn_integer(defaults[[nm]], nm)
  }
  for (nm in c("weight_draws", "noise_draws", "monitor_draws", "endpoint_draws")) {
    defaults[[nm]] <- .gltbnn_integer(defaults[[nm]], nm, minimum = 2L)
    if (defaults[[nm]] %% 2L != 0L) stop(nm, " must be even.", call. = FALSE)
  }
  for (nm in c("lr", "grad_clip")) {
    v <- defaults[[nm]]
    if (!is.numeric(v) || length(v) != 1L || !is.null(dim(v)) || !is.finite(v) || v <= 0) {
      stop(nm, " must be a positive finite number.", call. = FALSE)
    }
  }
  b <- defaults$logsd_bounds
  if (!is.numeric(b) || length(b) != 2L || !is.null(dim(b)) ||
      any(!is.finite(b)) || b[1L] >= b[2L]) {
    stop("logsd_bounds must contain two increasing finite numbers.", call. = FALSE)
  }
  defaults
}

# Restore .Random.seed even if the fit or prediction fails.
.gltbnn_with_seed <- function(seed, expr) {
  previous <- get0(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  on.exit({
    if (is.null(previous)) {
      if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    } else {
      assign(".Random.seed", previous, envir = .GlobalEnv)
    }
  })
  set.seed(seed)
  force(expr)
}
