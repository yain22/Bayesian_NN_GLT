# Fold-specific nuisance fits. Training data determine all preprocessing.
load_engine <- function() {
  engine <- new.env(parent = globalenv())
  for (file in list.files("package/R", "[.]R$", full.names = TRUE)) sys.source(file, engine)
  sys.source("engine-fit-batches.R", engine)
  engine
}

fold_sample <- function(data, fold, nuisance) {
  covariates <- c("age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown")
  x <- as.matrix(data[covariates])
  train <- data$fold != fold
  test <- which(!train)
  continuous <- nuisance != "ps"
  if (continuous) train <- train & data$e401 == as.integer(nuisance == "m1")
  raw <- x[train, , drop = FALSE]
  center <- colMeans(raw)
  spread <- apply(raw, 2L, sd)
  stopifnot(all(spread > 0), !any(train[test]))
  y <- data$e401[train] + 1L
  yc <- 0
  ys <- 1
  if (continuous) {
    yc <- mean(data$net_tfa[train])
    ys <- sd(data$net_tfa[train])
    y <- (data$net_tfa[train] - yc) / ys
  }
  list(x = scale(raw, center, spread), y = y, xtest = scale(x[test,, drop = FALSE], center, spread),
       family = if (continuous) "continuous" else "binary", test_id = data$row_id[test],
       y_center = yc, y_scale = ys)
}

batch_schedule <- function(n, seed, size = 128L, steps = 600L) {
  set.seed(seed)
  batches <- list()
  while (length(batches) < steps) {
    permutation <- sample.int(n)
    groups <- split(permutation, ceiling(seq_along(permutation) / size))
    batches <- c(batches, unname(groups))
  }
  batches[seq_len(steps)]
}

fnn_gradient <- function(x, y, weights, family, dropout, engine) {
  if (dropout) {
    h <- x
    features <- masks <- vector("list", length(weights))
    for (l in seq_along(weights)) {
      features[[l]] <- cbind(h, 1)
      z <- tcrossprod(features[[l]], weights[[l]])
      if (l < length(weights)) {
        mask <- matrix(rbinom(length(z), 1L, .8) / .8, nrow(z))
        h <- pmax(z, 0) * mask
        masks[[l]] <- (z > 0) * mask
      }
    }
    logits <- z
  } else {
    forward <- engine$.bnn_forward(x, weights)
    logits <- forward$logits
  }
  residual <- if (family == "binary") plogis(logits) - (y - 1L) else logits - y
  back <- residual / length(y)
  if (!dropout) return(engine$.bnn_backprop(back, forward, weights))
  gradient <- vector("list", length(weights))
  for (l in rev(seq_along(weights))) {
    gradient[[l]] <- crossprod(back, features[[l]])
    if (l > 1L) {
      w <- weights[[l]]
      back <- (back %*% w[, -ncol(w), drop = FALSE]) * masks[[l - 1L]]
    }
  }
  gradient
}

train_fnn <- function(sample, weights, batches, method, seed, engine) {
  set.seed(seed)
  first <- second <- lapply(weights, function(w) 0 * w)
  for (i in seq_along(batches)) {
    rows <- batches[[i]]
    gradient <- fnn_gradient(sample$x[rows,, drop = FALSE], sample$y[rows],
                             weights, sample$family, method == "Dropout", engine)
    for (l in seq_along(weights)) {
      g <- gradient[[l]]
      if (method == "L1") g <- g + .001 * sign(weights[[l]])
      first[[l]] <- .9 * first[[l]] + .1 * g
      second[[l]] <- .999 * second[[l]] + .001 * g^2
      weights[[l]] <- weights[[l]] - .003 * (first[[l]] / (1 - .9^i)) /
        (sqrt(second[[l]] / (1 - .999^i)) + 1e-8)
    }
  }
  weights
}

atomic_rds <- function(value, target) {
  temp <- tempfile(tmpdir = dirname(target))
  on.exit(unlink(temp))
  saveRDS(value, temp, version = 3L)
  if (!file.rename(temp, target)) stop("Could not commit ", basename(target))
}

fit_job <- function(job, data, engine, run_id) {
  sample <- fold_sample(data, job$fold, job$nuisance)
  hidden <- rep(as.integer(job$width), job$depth)
  seed <- 100000L * job$fold + 20000L * match(job$nuisance, c("m0", "m1", "ps")) +
    100L * job$depth + job$width
  key <- sprintf("f%d_%s_d%d_w%d", job$fold, job$nuisance, job$depth, job$width)
  initial_file <- file.path("initializers", paste0(key, ".rds"))
  if (job$method == "Initial") {
    weights <- engine$.bnn_initialize(sample$x, sample$y, hidden, seed, 0L, sample$family, 2L)
    batches <- batch_schedule(nrow(sample$x), seed + 1000L)
    weights <- train_fnn(sample, weights, batches, "Unpenalized", seed + 2000L, engine)
    atomic_rds(list(run_id = run_id, weights = weights), initial_file)
    return(TRUE)
  }
  initial <- readRDS(initial_file)
  stopifnot(identical(initial$run_id, run_id))
  batches <- batch_schedule(nrow(sample$x), seed + 3000L)
  if (job$method %in% c("G", "GL", "GLT")) {
    control <- engine$.gltbnn_control(list())
    fit <- engine$.gltbnn_engine_fit(sample$x, sample$y, hidden, job$method, control,
              seed + 4000L, outcome = sample$family, n_classes = 2L, geometry = "coefficient",
              initial_weights = initial$weights, batches = batches)
    stopifnot(fit$outer == 60L, nrow(fit$weight_steps) == 600L,
              all(is.finite(fit$weight_steps$sample_objective)))
    set.seed(seed + 5000L)
    prediction <- numeric(nrow(sample$xtest))
    for (i in seq_len(256L)) {
      weights <- engine$.bnn_draw_weights(fit$parameters, engine$.bnn_normal_draw(fit$parameters))
      value <- engine$.bnn_forward(sample$xtest, weights)$y
      prediction <- prediction + if (sample$family == "binary") plogis(value) else value
    }
    prediction <- prediction / 256
  } else {
    weights <- train_fnn(sample, initial$weights, batches, job$method, seed + 4000L, engine)
    prediction <- engine$.bnn_forward(sample$xtest, weights)$y
    if (sample$family == "binary") prediction <- plogis(prediction)
  }
  prediction <- sample$y_center + prediction * sample$y_scale
  stopifnot(all(is.finite(prediction)), length(prediction) == length(sample$test_id))
  if (sample$family == "binary") stopifnot(all(prediction >= 0 & prediction <= 1))
  atomic_rds(list(run_id = run_id, row_id = sample$test_id, prediction = prediction),
             file.path("checkpoints", paste0(key, "_", job$method, ".rds")))
  TRUE
}
