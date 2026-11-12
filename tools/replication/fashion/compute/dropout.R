# A separate Bernoulli mask for each training image and hidden activation.
# The backward pass follows the same masked activations as the forward pass.
fashion_dropout_gradient <- function(x, y, weights, rate, engine) {
  h <- x
  features <- vector("list", length(weights))
  derivative <- vector("list", length(weights) - 1L)
  for (l in seq_along(weights)) {
    features[[l]] <- cbind(h, 1)
    z <- features[[l]] %*% t(weights[[l]])
    if (l < length(weights)) {
      mask <- matrix(stats::rbinom(length(z), 1L, 1 - rate) / (1 - rate), nrow(z))
      h <- pmax(z, 0) * mask
      derivative[[l]] <- (z > 0) * mask
    }
  }
  log_probs <- engine$.bnn_class_log_probs(z, "binary")
  back <- -engine$.bnn_class_score(log_probs, y, "binary") / nrow(x)
  gradient <- vector("list", length(weights))
  for (l in rev(seq_along(weights))) {
    gradient[[l]] <- crossprod(back, features[[l]])
    if (l > 1L) {
      w <- weights[[l]]
      back <- (back %*% w[, -ncol(w), drop = FALSE]) * derivative[[l - 1L]]
    }
  }
  gradient
}

