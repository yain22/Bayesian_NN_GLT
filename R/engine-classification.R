# Columns follow the response levels. The single binary logit denotes level 2;
# the last categorical level is the reference and has logit zero.
.bnn_class_log_probs <- function(logits, outcome) {
  if (outcome == "binary") {
    eta <- logits[, 1L]
    return(cbind(stats::plogis(-eta, log.p = TRUE), stats::plogis(eta, log.p = TRUE)))
  }
  all_logits <- cbind(logits, 0)
  largest <- cbind(seq_len(nrow(logits)), max.col(all_logits, ties.method = "first"))
  shifted <- all_logits - all_logits[largest]
  # Separate the leading exp(0) so a nearly certain class retains its small
  # negative log probability even when 1 + sum(other terms) rounds to one.
  other_terms <- exp(shifted)
  other_terms[largest] <- 0
  shifted - log1p(rowSums(other_terms))
}

.bnn_class_score <- function(log_probs, y, outcome) {
  if (outcome == "binary") {
    # Evaluate the small complementary probability directly near a perfect fit.
    return(matrix(ifelse(y == 2L, exp(log_probs[, 1L]), -exp(log_probs[, 2L])), ncol = 1L))
  }
  last <- ncol(log_probs)
  score <- -exp(log_probs[, -last, drop = FALSE])
  rows <- which(y != last)
  observed <- cbind(rows, y[rows])
  score[observed] <- -expm1(log_probs[observed])
  score
}

.bnn_class_loglik <- function(log_probs, y) {
  sum(log_probs[cbind(seq_along(y), y)])
}

.gltbnn_engine_class_draws <- function(fit, x, draws, summarize = FALSE) {
  shape <- c(nrow(x), fit$config$n_classes)
  result <- array(0, dim = if (summarize) shape else c(shape, draws))
  for (b in seq_len(draws)) {
    weights <- .bnn_draw_weights(fit$parameters, .bnn_normal_draw(fit$parameters))
    logits <- .bnn_forward(x, weights)$logits
    if (any(!is.finite(logits))) {
      stop("Predicted logits exceed the finite numerical range. Check newdata and its predictor units.",
           call. = FALSE)
    }
    probabilities <- exp(.bnn_class_log_probs(logits, fit$config$outcome))
    if (summarize) {
      result <- result + probabilities
    } else {
      result[, , b] <- probabilities
    }
  }
  if (summarize) result / draws else result
}
