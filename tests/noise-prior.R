library(gltbnn)

x <- matrix(seq(-1, 1, length.out = 9), ncol = 1)
y <- as.vector(2 + 3 * x)
short <- list(init_steps = 3L, outer = 1L, weight_steps = 1L,
              weight_draws = 4L, noise_draws = 4L, monitor_draws = 4L, endpoint_draws = 4L)
prior <- c(shape = 3, scale = 2)

# The first weight block uses the common initializer and starting factors.
# The first variance coordinate then differs by exactly the two hyperparameters.
for (model in c("G", "GL", "GLT")) {
  default <- gltbnn_fit(x, y, 2L, model, short, 13L)
  explicit_default <- gltbnn_fit(x, y, 2L, model, short, 13L, noise_prior = NULL)
  proper <- gltbnn_fit(x, y, 2L, model, short, 13L, noise_prior = rev(prior))
  ref <- default$engine$state$noise
  q <- proper$engine$state$noise
  log_prior <- function(shape, scale, q) {
    shape * log(scale) - lgamma(shape) - (shape + 1) * q$zeta - scale * q$phi
  }
  expected <- log_prior(prior[["shape"]], prior[["scale"]], q) + q$entropy
  stopifnot(
    identical(default$engine$endpoint, explicit_default$engine$endpoint),
    identical(default$engine$parameters, proper$engine$parameters),
    is.null(default$noise_prior), identical(proper$noise_prior, prior),
    isTRUE(all.equal(q$a - ref$a, prior[["shape"]])),
    isTRUE(all.equal(q$b - ref$b, prior[["scale"]])),
    isTRUE(all.equal(proper$engine$endpoint$components[["noise_neg_kl"]], expected)),
    isTRUE(all.equal(proper$engine$components$noise_neg_kl[2L], expected)),
    all(proper$engine$components$noise_neg_kl <= 0),
    is.finite(proper$engine$endpoint$value),
    all(is.finite(predict(proper, x, type = "response_draws", draws = 8L)))
  )
}

# High-precision reference KL for a prior and a fitted factor separated
# by only a few representable shape and scale increments.
concentrated <- gltbnn_fit(
  matrix(c(-1, 0, 1, 2), ncol = 1L), c(2, -1, 1, 0), hidden = 1L, prior = "G",
  seed = 1L, noise_prior = c(shape = 1e16, scale = 1e16),
  control = list(init_steps = 1L, outer = 1L, weight_steps = 1L)
)
endpoint <- concentrated$engine$endpoint
expected_kl <- 1.99999999999999826667e-16
stopifnot(abs(-endpoint$components[["noise_neg_kl"]] / expected_kl - 1) < 1e-12,
          all(concentrated$engine$components$noise_neg_kl <= 0),
          isTRUE(all.equal(sum(endpoint$components), endpoint$value)))

for (bad in list(c(3, 2), c(shape = 3, rate = 2), c(shape = 2, shape = 3),
                 c(shape = 0, scale = 1), c(shape = 1, scale = -1),
                 c(shape = NA, scale = 1), c(shape = 1, scale = Inf),
                 matrix(prior, 1L), list(shape = 3, scale = 2))) {
  msg <- tryCatch(gltbnn_fit(x, y, 2L, "G", short, 13L, noise_prior = bad), error = conditionMessage)
  stopifnot(is.character(msg), grepl("noise_prior", msg, fixed = TRUE))
}
