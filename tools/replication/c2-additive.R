# Reproduce one matched comparison in the Gaussian simulation.
# With gltbnn 0.2.x installed, run this command from the repository root.
#   Rscript --vanilla tools/replication/c2-additive.R
# Or source this file and run result <- run_c2_replication().
# To regenerate the fitted-neuron panel data from the same run:
#   Rscript --vanilla tools/replication/c2-additive.R --write-neuron-scales
#
# C2, training SNR 3, replicate 1, two hidden layers of width 10.
# Each method uses the same 300 training cases, 2,000 test cases and starting
# weights. Test RMSE measures error against Y; mean RMSE uses the known signal.
# Coverage refers to central 90% pointwise intervals on these 2,000 cases.
# Paper summaries average over 100 independent replicates.
# For this replicate the archived test RMSEs, rounded to six decimal places,
# are G 1.000002, GL 0.991699, GLT 1.116787, FNN 1.425569, dropout 1.151745
# and L1 1.145793. Different math libraries can change floating-point results.

study_seed <- function(...) {
  label <- paste(..., collapse = "|")
  hash <- 20261030
  for (byte in utf8ToInt(label)) hash <- (131 * hash + byte) %% 2147483646
  as.integer(hash + 1)
}

ar1_predictors <- function(n, p, rho) {
  x <- matrix(rnorm(n * p), n, p)
  for (j in 2:p) x[, j] <- rho * x[, j - 1L] + sqrt(1 - rho^2) * x[, j]
  colnames(x) <- paste0("x", seq_len(p))
  x
}

c2_data <- function() {
  active <- 10L
  rho <- 0.3
  signal <- function(x) {
    z <- x[, seq_len(active), drop = FALSE]
    rowSums(sin(z) + .5 * (z^2 - 1)) / sqrt(active)
  }
  # Standardize the signal against a separate reference population.
  set.seed(study_seed("reference", "additive", active, rho, 100000L))
  reference <- signal(ar1_predictors(100000L, active, rho))
  center <- mean(reference)
  scale <- sd(reference)
  sample_split <- function(n, split) {
    set.seed(study_seed("cavi100_v1", 2L, 1L, "gaussian", split))
    x <- ar1_predictors(n, 100L, rho)
    list(x = x, truth = (signal(x) - center) / scale, noise = rnorm(n))
  }
  train <- sample_split(300L, "train")
  test <- sample_split(2000L, "test")
  # Both splits use the scale estimated from the training draw alone.
  error_sd <- sqrt(var(train$truth) / (3 * var(train$noise)))
  train$y <- train$truth + error_sd * train$noise
  test$y <- test$truth + error_sd * test$noise
  stopifnot(abs(var(train$truth) / var(train$y - train$truth) - 3) < 1e-12)
  list(train = train, test = test)
}

forward_pass <- function(x, weights, dropout) {
  h <- x
  inputs <- preactivation <- masks <- vector("list", length(weights))
  for (l in seq_along(weights)) {
    inputs[[l]] <- cbind(h, 1)
    z <- inputs[[l]] %*% t(weights[[l]])
    preactivation[[l]] <- z
    if (l == length(weights)) {
      h <- z
    } else {
      h <- pmax(z, 0)
      mask <- matrix(1, nrow(h), ncol(h))
      if (dropout > 0) {
        mask <- matrix(rbinom(length(h), 1, 1 - dropout) / (1 - dropout), nrow(h))
      }
      masks[[l]] <- mask
      h <- h * mask
    }
  }
  list(y = as.vector(h), inputs = inputs, preactivation = preactivation, masks = masks)
}

# Full-batch Adam for the warm start, FNN, dropout and L1 comparators.
# Each call starts new Adam moments. The L1 penalty includes the biases.
train_network <- function(x, y, hidden, seed, weights = NULL, dropout = 0, l1 = 0) {
  set.seed(seed)
  if (is.null(weights)) {
    dimensions <- c(ncol(x), hidden) + 1L
    widths <- c(hidden, 1L)
    weights <- lapply(seq_along(widths), function(l) {
      matrix(rnorm(widths[l] * dimensions[l], sd = sqrt(2 / dimensions[l])), widths[l])
    })
  }
  first <- second <- lapply(weights, function(w) w * 0)
  for (step in seq_len(600L)) {
    pass <- forward_pass(x, weights, dropout)
    residual <- matrix((pass$y - y) / length(y), ncol = 1L)
    gradient <- vector("list", length(weights))
    for (l in rev(seq_along(weights))) {
      gradient[[l]] <- crossprod(residual, pass$inputs[[l]]) + l1 * sign(weights[[l]])
      if (l > 1L) {
        slopes <- weights[[l]][, -ncol(weights[[l]]), drop = FALSE]
        residual <- (residual %*% slopes) * (pass$preactivation[[l - 1L]] > 0) * pass$masks[[l - 1L]]
      }
    }
    for (l in seq_along(weights)) {
      first[[l]] <- .9 * first[[l]] + .1 * gradient[[l]]
      second[[l]] <- .999 * second[[l]] + .001 * gradient[[l]]^2
      weights[[l]] <- weights[[l]] - .003 * (first[[l]] / (1 - .9^step)) /
        (sqrt(second[[l]] / (1 - .999^step)) + 1e-8)
    }
  }
  weights
}

interval_summary <- function(draws, target) {
  bounds <- apply(draws, 1L, quantile, probs = c(.05, .95), names = FALSE)
  c(coverage = mean(target >= bounds[1L, ] & target <= bounds[2L, ]),
    width = mean(bounds[2L, ] - bounds[1L, ]))
}

first_layer_summary <- function(fit, coefficients = gltbnn::gltbnn_shrinkage(fit)) {
  coefficients <- subset(coefficients, layer == 1L & !is_bias)
  log_sd <- log(coefficients$geometric_prior_sd)
  spreads <- vapply(split(log_sd, coefficients$unit), function(x) {
    diff(quantile(x, c(.1, .9), names = FALSE))
  }, numeric(1))
  c(active_median_log_sd_q = median(log_sd[coefficients$coordinate <= 10L]),
    inactive_median_log_sd_q = median(log_sd[coefficients$coordinate > 10L]),
    within_unit_log_ratio = median(spreads))
}

run_c2_replication <- function() {
  if (!requireNamespace("gltbnn", quietly = TRUE) ||
      packageVersion("gltbnn") < "0.2.0" || packageVersion("gltbnn") >= "0.3.0") {
    stop("Install gltbnn 0.2.x to run this replication.")
  }
  previous_kind <- RNGkind()
  previous_seed <- get0(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  on.exit({
    do.call(RNGkind, as.list(previous_kind))
    if (is.null(previous_seed)) {
      rm(".Random.seed", envir = .GlobalEnv)
    } else {
      assign(".Random.seed", previous_seed, envir = .GlobalEnv)
    }
  })
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  # The simulation pairs a pretrained network across all six fits and uses a
  # separate random stream for variational inference. This version-specific
  # adapter exposes the same Gaussian coefficient engine used by gltbnn_fit()
  # and passes the exact common starting matrices to it.
  fit_engine <- getFromNamespace(".gltbnn_engine_fit", "gltbnn")
  fit_control <- getFromNamespace(".gltbnn_control", "gltbnn")
  controls <- fit_control(list(outer = 60L, weight_steps = 10L, init_steps = 600L,
    lr = .003, weight_draws = 4L, noise_draws = 16L, monitor_draws = 64L,
    endpoint_draws = 256L, grad_clip = 10, logsd_bounds = c(-15, 3), max_backtracks = 10L))

  data <- c2_data()
  train <- data$train
  test <- data$test
  preprocessing <- list(x_center = colMeans(train$x), x_scale = apply(train$x, 2L, sd),
                        y_center = mean(train$y), y_scale = sd(train$y))
  scale_x <- function(x) sweep(sweep(x, 2L, preprocessing$x_center, "-"), 2L, preprocessing$x_scale, "/")
  x <- scale_x(train$x)
  y <- (train$y - preprocessing$y_center) / preprocessing$y_scale
  hidden <- c(10L, 10L)
  initial_seed <- study_seed("cavi100_v1", 2L, 1L, "initial")
  vi_seed <- study_seed("relu100_v1", 2L, 1L, "variational")
  pred_seed <- study_seed(initial_seed, "predict")
  shared_weights <- train_network(x, y, hidden, initial_seed)
  methods <- c(G = "g", GL = "hs", GLT = "glt", FNN = "fnn", Dropout = "dropout", L1 = "l1")
  predictions <- intervals <- shrinkage <- elbo <- coefficient_scales <- list()
  for (method in names(methods)) {
    if (method %in% c("G", "GL", "GLT")) {
      engine <- fit_engine(x, y, hidden, method, controls, vi_seed, initial_weights = shared_weights)
      stopifnot(engine$config$geometry == "coefficient", engine$outer == 60L,
                nrow(engine$weight_steps) == 600L)
      fit <- structure(list(prior = method, outcome = "continuous", geometry = "coefficient",
        hidden = hidden, p = ncol(x), preprocess = preprocessing, predictor_names = colnames(train$x),
        engine = engine), class = "gltbnn_fit")
      # The study uses separate streams for posterior network draws and
      # observation errors. One noise variance is drawn per network draw.
      stream_seed <- function(offset) as.integer((as.double(pred_seed) + offset) %% 2147483646 + 1)
      latent <- predict(fit, test$x, type = "mean_draws", draws = 256L, seed = stream_seed(0))
      predictions[[method]] <- rowMeans(latent)
      set.seed(stream_seed(104729))
      noise <- engine$state$noise
      noise_sd <- preprocessing$y_scale / sqrt(rgamma(256L, shape = noise$a, rate = noise$b))
      errors <- matrix(rnorm(length(latent)), nrow = 256L) * noise_sd
      response <- latent + t(errors)
      intervals[[method]] <- c(setNames(interval_summary(latent, test$truth), c("mean_coverage", "mean_width")),
        setNames(interval_summary(response, test$y), c("response_coverage", "response_width")))
      coefficients <- gltbnn::gltbnn_shrinkage(fit)
      shrinkage[[method]] <- first_layer_summary(fit,coefficients)
      coefficient_scales[[method]] <- coefficients[c("layer","unit","coordinate","is_bias","geometric_prior_sd")]
      elbo[[method]] <- unlist(engine$endpoint[c("value", "mcse")])
    } else {
      w <- train_network(x, y, hidden, study_seed(initial_seed, methods[[method]]), shared_weights,
                         dropout = if (method == "Dropout") .2 else 0, l1 = if (method == "L1") .001 else 0)
      predictions[[method]] <- forward_pass(scale_x(test$x), w, dropout = 0)$y *
        preprocessing$y_scale + preprocessing$y_center
    }
  }
  scores <- t(vapply(predictions, function(mu) {
    c(rmse = sqrt(mean((mu - test$y)^2)), mean_rmse = sqrt(mean((mu - test$truth)^2)))
  }, numeric(2)))
  list(scores = scores, intervals = do.call(rbind, intervals), shrinkage = do.call(rbind, shrinkage),
       endpoint_elbo = do.call(rbind, elbo), predictions = predictions,
       coefficient_scales = coefficient_scales)
}

write_neuron_scales <- function(result, file = "docs/study/c2-scales.js") {
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Install jsonlite to export the panel.")
  scales <- result$coefficient_scales
  stopifnot(identical(names(scales),c("G","GL","GLT")))
  layers <- lapply(1:2,function(layer) {
    dimension <- c(101L,11L)[layer]
    lapply(1:10,function(unit) {
      lapply(scales,function(s) {
        rows <- s[s$layer == layer & s$unit == unit,]
        stopifnot(identical(rows$coordinate,seq_len(dimension)),
                  identical(which(rows$is_bias),dimension),
                  all(is.finite(rows$geometric_prior_sd) & rows$geometric_prior_sd > 0))
        rows$geometric_prior_sd
      })
    })
  })
  json <- jsonlite::toJSON(layers,digits=15)
  writeBin(charToRaw(paste0("const c2NeuronScales = ",json,";\n")),file)
}

if (sys.nframe() == 0L) {
  result <- run_c2_replication()
  if ("--write-neuron-scales" %in% commandArgs(trailingOnly = TRUE)) write_neuron_scales(result)
  for (name in c("scores", "intervals", "shrinkage", "endpoint_elbo")) {
    cat("\n", name, "\n", sep = "")
    print(result[[name]], digits = 7L)
  }
}
