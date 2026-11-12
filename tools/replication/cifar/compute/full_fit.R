# Matched full-split CIFAR-10 cat and dog experiment. Run from the study directory.
cifar_hash <- function(x) digest::digest(x, algo = "md5", serializeVersion = 2L)

cifar_engine <- function() {
  e <- new.env(parent = globalenv())
  for (f in list.files("package/R", pattern = "\\.[Rr]$", full.names = TRUE)) sys.source(f, e)
  sys.source("compute/engine-fit-batches.R", e)
  e
}

cifar_batches <- function(n, batch_size, steps, seed) {
  set.seed(seed)
  batches <- list()
  while (length(batches) < steps) {
    order <- sample.int(n)
    groups <- split(order, ceiling(seq_along(order) / batch_size))
    batches <- c(batches, unname(groups))
  }
  batches[seq_len(steps)]
}

cifar_prepare <- function(data) {
  stopifnot(is.raw(data$Xtrain), is.raw(data$Xtest),
            identical(dim(data$Xtrain), c(10000L, 3072L)),
            identical(dim(data$Xtest), c(2000L, 3072L)),
            length(data$train_index) == 10000L, length(data$test_index) == 2000L,
            !is.unsorted(data$train_index, strictly = TRUE), !is.unsorted(data$test_index, strictly = TRUE),
            all(data$train_index %in% seq_len(50000L)), all(data$test_index %in% seq_len(10000L)),
            identical(tabulate(data$ytrain + 1L, 2L), rep(5000L, 2L)),
            identical(tabulate(data$ytest + 1L, 2L), rep(1000L, 2L)),
            all(data$ytrain %in% 0:1), all(data$ytest %in% 0:1))
  center <- spread <- numeric(ncol(data$Xtrain))
  for (j in seq_len(ncol(data$Xtrain))) {
    pixel <- as.numeric(data$Xtrain[, j]) / 255
    center[j] <- mean(pixel)
    spread[j] <- stats::sd(pixel)
  }
  keep <- which(spread > 0)
  stopifnot(identical(keep, seq_len(3072L)))
  transform <- function(raw) {
    x <- matrix(0, nrow(raw), length(keep))
    for (k in seq_along(keep)) {
      j <- keep[k]
      x[, k] <- (as.numeric(raw[, j]) / 255 - center[j]) / spread[j]
    }
    x
  }
  list(x = transform(data$Xtrain), xtest = transform(data$Xtest),
       y = as.integer(data$ytrain) + 1L, truth = as.integer(data$ytest),
       train_id = data$train_index, test_id = data$test_index,
       preprocess = list(x_center = center, x_scale = spread, x_keep = keep))
}

cifar_fnn_update <- function(data, weights, batches, method, engine, seed) {
  set.seed(seed)
  first <- second <- lapply(weights, function(w) w * 0)
  for (step in seq_along(batches)) {
    ids <- batches[[step]]
    x <- data$x[ids, , drop = FALSE]
    y <- data$y[ids]
    if (method == "Dropout") {
      gradient <- cifar_dropout_gradient(x, y, weights, .2, engine)
    } else {
      forward <- engine$.bnn_forward(x, weights)
      lp <- engine$.bnn_class_log_probs(forward$logits, "binary")
      back <- -engine$.bnn_class_score(lp, y, "binary") / length(ids)
      gradient <- engine$.bnn_backprop(back, forward, weights)
    }
    for (l in seq_along(weights)) {
      g <- gradient[[l]]
      if (method == "L1") g <- g + .001 * sign(weights[[l]])
      first[[l]] <- .9 * first[[l]] + .1 * g
      second[[l]] <- .999 * second[[l]] + .001 * g^2
      weights[[l]] <- weights[[l]] - .003 * (first[[l]] / (1 - .9^step)) /
        (sqrt(second[[l]] / (1 - .999^step)) + 1e-8)
    }
  }
  stopifnot(all(vapply(weights, function(w) all(is.finite(w)), logical(1L))))
  weights
}

cifar_save <- function(value, path) {
  if (file.exists(path)) stop("Existing checkpoint: ", basename(path))
  temporary <- tempfile("pending-", tmpdir = dirname(path))
  on.exit(unlink(temporary))
  saveRDS(value, temporary, version = 3L)
  if (!file.rename(temporary, path)) stop("Could not save ", basename(path))
}

cifar_init_path <- function(depth, width) sprintf("initializers/d%d_w%d.rds", depth, width)

cifar_warmup <- function(data, depth, width, e) {
  seed <- as.integer(241000L + 100L * depth + width)
  hidden <- rep(as.integer(width), depth)
  batches <- cifar_batches(nrow(data$x), 1024L, 600L, seed + 40000L)
  stopifnot(all(tabulate(unlist(batches), nrow(data$x)) == 60L))
  initial <- e$.bnn_initialize(data$x, data$y, hidden, seed, 0L, "binary", 2L)
  warmed <- cifar_fnn_update(data, initial, batches, "Unpenalized", e, seed)
  list(seed = seed, weights = warmed, initialization_md5 = cifar_hash(warmed),
       warmup_md5 = cifar_hash(batches))
}

# All test rows see the same joint draw. Use log sums for the observed class
# so a very confident error can be scored even when its probability underflows.
cifar_predict <- function(x, truth, e, weights = NULL, parameters = NULL, draws = 256L,
                          seed = 1L, chunk_size = 1024L) {
  posterior <- !is.null(parameters)
  stopifnot(xor(posterior, !is.null(weights)))
  if (!posterior) draws <- 1L
  rows <- unname(split(seq_len(nrow(x)), ceiling(seq_len(nrow(x)) / chunk_size)))
  augmented <- lapply(rows, function(i) cbind(x[i, , drop = FALSE], 1))
  probability <- matrix(0, nrow(x), 2L, dimnames = list(NULL, c("cat", "dog")))
  true_logsum <- rep(-Inf, nrow(x))
  if (posterior) set.seed(seed)
  for (b in seq_len(draws)) {
    if (posterior) weights <- e$.bnn_draw_weights(parameters, e$.bnn_normal_draw(parameters))
    for (k in seq_along(rows)) {
      i <- rows[[k]]
      h <- augmented[[k]]
      for (layer in seq_along(weights)) {
        z <- h %*% t(weights[[layer]])
        if (layer < length(weights)) h <- cbind(pmax(z, 0), 1)
      }
      lp <- e$.bnn_class_log_probs(z, "binary")
      stopifnot(all(is.finite(lp)))
      probability[i, ] <- probability[i, ] + exp(lp)
      observed <- lp[cbind(seq_along(i), truth[i] + 1L)]
      if (b == 1L) {
        true_logsum[i] <- observed
      } else {
        old <- true_logsum[i]
        true_logsum[i] <- pmax(old, observed) + log1p(exp(-abs(old - observed)))
      }
    }
  }
  prob <- probability / draws
  true_logprob <- true_logsum - log(draws)
  pred <- max.col(prob, ties.method = "first") - 1L
  stopifnot(max(abs(rowSums(prob)-1)) < 1e-10)
  list(prob = prob, true_logprob = true_logprob, pred = pred,
       accuracy = mean(pred == truth), log_loss = -mean(true_logprob))
}

cifar_fit <- function(data, depth, width, method, init, e) {
  start <- proc.time()[["elapsed"]]
  seed <- as.integer(241000L + 100L * depth + width)
  hidden <- rep(as.integer(width), depth)
  stopifnot(identical(init$seed, seed), identical(init$initialization_md5, cifar_hash(init$weights)))
  batches <- cifar_batches(nrow(data$x), 1024L, 600L, seed + 50000L)
  stopifnot(all(tabulate(unlist(batches), nrow(data$x)) == 60L))
  audit <- list(batch_size = 1024L, warmup_seed = seed + 40000L, continuation_seed = seed + 50000L,
                steps = 600L, warmup_md5 = init$warmup_md5, continuation_md5 = cifar_hash(batches),
                initialization_md5 = init$initialization_md5)
  record <- list(method = method, depth = depth, width = width, seed = seed, schedule = audit,
                 test_id = data$test_id, truth = data$truth, train_id = data$train_id,
                 n_train = nrow(data$x), levels = c("cat", "dog"), reference = "cat", positive = "dog",
                 retained_inputs = data$preprocess$x_keep, preprocess = data$preprocess)
  if (method %in% c("G", "GL", "GLT")) {
    ctl <- e$.gltbnn_control(list())
    fit <- e$.gltbnn_engine_fit(data$x, data$y, hidden, method, ctl, seed,
                 outcome = "binary", n_classes = 2L, geometry = "coefficient",
                 initial_weights = init$weights, batches = batches)
    stopifnot(is.null(fit$state$noise), fit$outer == 60L,
              all(is.finite(fit$weight_steps$sample_objective)),
              all(is.finite(fit$weight_steps$sample_gain)), nrow(fit$weight_steps) == 600L)
    record <- c(record, fit[c("parameters","control","trace","weight_steps","config","endpoint")],
                list(predict_seed = seed + 10000L, predict_draws = 256L))
    prediction <- cifar_predict(data$xtest, data$truth, e, parameters = fit$parameters,
                                draws = 256L, seed = seed + 10000L)
  } else {
    weights <- cifar_fnn_update(data, init$weights, batches, method, e, seed + 30000L)
    record <- c(record, list(weights = weights, init_steps = 600L, steps = 600L,
           dropout = if (method == "Dropout") .2 else 0,
           l1 = if (method == "L1") .001 else 0, dropout_seed = seed + 30000L))
    prediction <- cifar_predict(data$xtest, data$truth, e, weights = weights)
  }
  c(record, prediction, list(elapsed_seconds = proc.time()[["elapsed"]] - start))
}
