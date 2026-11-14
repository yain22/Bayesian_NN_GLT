library(gltbnn)
d <- gltbnn::sipp401k
vars <- c("age", "inc", "educ", "fsize", "marr", "twoearn",
          "db", "pira", "hown")
x <- as.matrix(d[vars])
y <- d$net_tfa
a <- d$e401

K <- 5L
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(401015L)
fold <- integer(length(a))
offset <- 0L
for (group in 0:1) {
  rows <- sample(which(a == group))
  fold[rows] <- 1L + (seq_along(rows) - 1L + offset) %% K
  offset <- (offset + length(rows)) %% K
}

fit_fold <- function(x, y, a, fold, k, hidden, control) {
  train <- fold != k
  test <- which(!train)
  arm0 <- train & a == 0L
  arm1 <- train & a == 1L
  s <- 100000L * k + 100L * length(hidden) + hidden[1L] +
    20000L * (1:3)

  fit0 <- gltbnn_fit(x[arm0, ], y[arm0], outcome = "continuous",
                     hidden = hidden, prior = "GLT",
                     control = control, seed = s[1L])
  fit1 <- gltbnn_fit(x[arm1, ], y[arm1], outcome = "continuous",
                     hidden = hidden, prior = "GLT",
                     control = control, seed = s[2L])
  eligible <- factor(a[train], levels = 0:1,
                     labels = c("ineligible", "eligible"))
  fit_e <- gltbnn_fit(x[train, ], eligible, outcome = "binary",
                      hidden = hidden, prior = "GLT",
                      control = control, seed = s[3L])

  newx <- x[test, , drop = FALSE]
  m0 <- predict(fit0, newx, type = "mean", draws = 256L,
                seed = s[1L] + 5000L)
  m1 <- predict(fit1, newx, type = "mean", draws = 256L,
                seed = s[2L] + 5000L)
  prob <- predict(fit_e, newx, type = "prob", draws = 256L,
                  seed = s[3L] + 5000L)
  data.frame(row = test, m0 = m0, m1 = m1, e = prob[, "eligible"])
}

hidden <- rep(25L, 10L)
control <- list(batch_size = 128L, init_steps = 600L,
                outer = 60L, weight_steps = 10L)
m0 <- m1 <- e <- rep(NA_real_, length(y))
for (k in seq_len(K)) {
  p <- fit_fold(x, y, a, fold, k, hidden, control)
  m0[p$row] <- p$m0
  m1[p$row] <- p$m1
  e[p$row] <- p$e
}

e_bound <- pmin(0.99, pmax(0.01, e))
psi <- m1 - m0 + a * (y - m1) / e_bound -
  (1 - a) * (y - m0) / (1 - e_bound)
ate <- mean(psi)
se <- sqrt(mean((psi - ate)^2) / length(psi))
ci <- ate + c(-1, 1) * qnorm(0.975) * se
round(c(estimate = ate, SE = se, lower = ci[1L], upper = ci[2L]))
