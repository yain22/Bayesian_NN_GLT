source("reproduce_401k.R")
d <- data_401k()
hidden <- rep(25L, 10L)
score <- rep(NA_real_, nrow(d))

for (k in 1:5) {
  i <- which(d$fold == k)
  p <- fit_401k_glt_fold(d, k, hidden = hidden)
  stopifnot(identical(p$row_id, d$row_id[i]))
  a <- d$e401[i]
  y <- d$net_tfa[i]
  e <- pmin(0.99, pmax(0.01, p$e))
  score[i] <- p$m1 - p$m0 + a * (y - p$m1) / e -
    (1 - a) * (y - p$m0) / (1 - e)
}

ate <- mean(score)
se <- sqrt(mean((score - ate)^2) / length(score))
round(c(estimate = ate, SE = se,
        lower = ate - qnorm(0.975) * se,
        upper = ate + qnorm(0.975) * se))
