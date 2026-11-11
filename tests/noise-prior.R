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
  expected <- log_prior(prior[["shape"]], prior[["scale"]], q)
  stopifnot(
    identical(default$engine$endpoint, explicit_default$engine$endpoint),
    identical(default$engine$parameters, proper$engine$parameters),
    is.null(default$noise_prior), identical(proper$noise_prior, prior),
    isTRUE(all.equal(q$a - ref$a, prior[["shape"]])),
    isTRUE(all.equal(q$b - ref$b, prior[["scale"]])),
    isTRUE(all.equal(proper$engine$endpoint$components[["noise_prior"]], expected)),
    isTRUE(all.equal(proper$engine$components$noise_prior[2L], expected)),
    is.finite(proper$engine$endpoint$value),
    all(is.finite(predict(proper, x, type = "response_draws", draws = 8L)))
  )
}

for (bad in list(c(3, 2), c(shape = 3, rate = 2), c(shape = 2, shape = 3),
                 c(shape = 0, scale = 1), c(shape = 1, scale = -1),
                 c(shape = NA, scale = 1), c(shape = 1, scale = Inf),
                 matrix(prior, 1L), list(shape = 3, scale = 2))) {
  msg <- tryCatch(gltbnn_fit(x, y, 2L, "G", short, 13L, noise_prior = bad), error = conditionMessage)
  stopifnot(is.character(msg), grepl("noise_prior", msg, fixed = TRUE))
}
