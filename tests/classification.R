library(gltbnn)

# Both standardization and a later forward pass can overflow even when all
# supplied predictor values are finite. Check this through the public method.
small_x <- matrix((1:6) * 1e-90, ncol = 1L)
tiny_control <- list(init_steps=1L, outer=1L, weight_steps=1L,
                     weight_draws=2L, monitor_draws=2L, endpoint_draws=2L)
for (family in c("continuous", "binary", "categorical")) {
  outcomes <- switch(family, continuous=1:6, binary=rep(0:1,3L), categorical=rep(0:2,2L))
  f <- gltbnn_fit(small_x, outcomes, hidden=2L, prior="G", outcome=family, control=tiny_control)
  overflow <- tryCatch(predict(f, matrix(1e230), draws=2L), error=conditionMessage)
  stopifnot(is.character(overflow), grepl("transformation overflows", overflow, fixed=TRUE))
  f$engine$parameters$mu[[1L]][] <- 1e200
  f$engine$parameters$mu[[2L]][] <- 1e200
  kind <- if (family=="continuous") "mean" else "prob"
  overflow <- tryCatch(predict(f, matrix(1e20), type=kind, draws=2L), error=conditionMessage)
  stopifnot(is.character(overflow), grepl("finite numerical range", overflow, fixed=TRUE))
}

x <- data.frame(a = c(-1, 2, 0, 1, -2, 0), constant = 0,
                b = c(0, 1, -2, 2, -1, 1))
binary <- factor(c("yes", "no", "yes", "no", "yes", "no"),
                 levels = c("yes", "no"))
multi <- c(10L, 2L, 3L, 2L, 10L, 3L)
short <- list(init_steps = 4L, outer = 2L, weight_steps = 1L,
              weight_draws = 4L, monitor_draws = 4L, endpoint_draws = 4L)

set.seed(7)
rng <- .Random.seed
for (family in c("binary", "categorical")) {
  y <- if (family == "binary") binary else multi
  classes <- if (family == "binary") c("yes", "no") else c("2", "3", "10")
  for (prior in c("G", "GL", "GLT")) {
    fit <- gltbnn_fit(x, y, hidden = c(3, 2), prior = prior,
                      outcome = family, control = short)
    p <- predict(fit, x, type = "prob", draws = 5L)
    s <- predict(fit, x, type = "prob_draws", draws = 5L)
    predicted <- predict(fit, x, draws = 5L)
    coefficients <- gltbnn_shrinkage(fit)
    stopifnot(identical(fit$levels, classes), identical(fit$outcome, family),
              is.null(fit$engine$state$noise), is.finite(fit$engine$endpoint$value),
              identical(fit$geometry, "coefficient"), nrow(coefficients) == 17L,
              all(is.finite(coefficients$geometric_prior_sd)),
              all(coefficients$geometric_prior_sd > 0), all(coefficients$group_size == 1L),
              identical(coefficients$unit, c(rep(1:3, each = 3L), rep(1:2, each = 4L))),
              identical(coefficients$coordinate, c(rep(1:3, 3L), rep(1:4, 2L))),
              identical(which(coefficients$is_bias), c(3L, 6L, 9L, 13L, 17L)),
              identical(dim(s), c(nrow(x), length(classes), 5L)),
              identical(colnames(p), classes),
              isTRUE(all.equal(p, rowMeans(s, dims = 2), tolerance = 1e-14)),
              identical(levels(predicted), classes),
              identical(as.integer(predicted), max.col(p, ties.method = "first")),
              max(abs(rowSums(p) - 1)) < 1e-14,
              max(abs(apply(s, c(1, 3), sum) - 1)) < 1e-14, all(s >= 0 & s <= 1))

    # Identify each scalar by its (receiving unit, incoming coordinate), including
    # the bias in the final matrix column, independently of the printed row order.
    for (i in seq_len(nrow(coefficients))) {
      row <- coefficients[i, ]
      layer <- row$layer
      g <- fit$engine$state$g[[layer]]
      idx <- row$unit + (row$coordinate - 1L) * fit$hidden[layer]
      precision <- switch(prior, G = g$tau$phi, GL = g$tau$phi * g$local$phi[idx],
                          GLT = g$local$phi[idx])
      log_sd <- switch(prior, G = g$tau$zeta / 2, GL = (g$tau$zeta + g$local$zeta[idx]) / 2,
                       GLT = g$local$zeta[idx])
      stopifnot(identical(row$scale_precision, precision),
                isTRUE(all.equal(row$geometric_prior_sd, exp(log_sd))))
    }

    # A column constant in training stays excluded even when it varies at prediction.
    newx <- x[, c("constant", "b", "a")]
    newx$constant <- 1:6
    stopifnot(identical(p, predict(fit, newx, type = "prob", draws = 5L)),
              identical(dim(predict(fit, x[1, ], type = "prob_draws", draws = 1L)),
                        c(1L, length(classes), 1L)),
              identical(dim(predict(fit, x[1, ], type = "prob", draws = 1L)),
                        c(1L, length(classes))))

    # Known logits identify the modeled event and the softmax reference level.
    # Fix only the prediction fixture's readout; training above uses the fitted model.
    last <- length(fit$engine$parameters$mu)
    logits <- if (family == "binary") log(3) else log(c(3, 2))
    readout <- fit$engine$parameters$mu[[last]]
    readout[] <- 0
    readout[, ncol(readout)] <- logits
    fit$engine$parameters$mu[[last]] <- readout
    fit$engine$parameters$logsd[[last]][] <- -100
    expected <- if (family == "binary") c(1, 3) / 4 else c(3, 2, 1) / 6
    stopifnot(max(abs(predict(fit, x[1, ], type = "prob") - expected)) < 1e-14)
  }
}
stopifnot(identical(.Random.seed, rng))

fails <- function(expr, pattern) {
  error <- tryCatch({force(expr); NULL}, error = conditionMessage)
  is.character(error) && grepl(pattern, error, fixed = TRUE)
}
stopifnot(fails(gltbnn_fit(x, multi, outcome = "binary"), "exactly two"),
          fails(gltbnn_fit(x, rep("one", 6), outcome = "categorical"), "at least two"),
          fails(gltbnn_fit(x, factor(binary, levels = c("no", "yes", "other")),
                           outcome = "categorical"), "unused factor levels"),
          fails(gltbnn_fit(x, replace(multi, 1, NA), outcome = "categorical"), "nonmissing"),
          fails(gltbnn_fit(x, binary, outcome = "binary",
                           noise_prior = c(shape = 3, scale = 2)), "noise_prior"),
          fails(gltbnn_fit(x["constant"], binary, outcome = "binary"), "varying"),
          fails(predict(fit, x, type = "mean"), "arg"),
          fails(predict(fit, x[-1L]), "number of predictors"))

# Serialized regression fits from the previous release lack an outcome field.
gaussian <- gltbnn_fit(x[c("a", "b")], c(-1, 0.1, 1, -0.3, 0.4, 0.9),
                       hidden = 2, prior = "G", control = short)
before <- predict(gaussian, x[c("a", "b")], draws = 4L)
gaussian$outcome <- NULL
stopifnot(identical(before, predict(gaussian, x[c("a", "b")], draws = 4L)))
