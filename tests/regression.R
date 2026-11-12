library(gltbnn)

x <- cbind(u = c(-2, -1, -0.5, 0, 0.25, 0.8, 1, 2),
           v = c(1, 0, 2, -1, 0, 3, -2, 1))
y <- 7 + x[, 1] * x[, 2] + 0.5 * x[, 1]
newx <- x[c(1, 4, 8), , drop = FALSE]
control <- list(init_steps = 4L, outer = 2L, weight_steps = 2L,
                weight_draws = 4L, noise_draws = 4L, monitor_draws = 4L,
                endpoint_draws = 4L)

# Values from the manuscript's reference implementation for this small network.
# The columns give the endpoint ELBO, standardized noise precision, and the
# three posterior means on the response scale (32 common weight draws).
reference <- rbind(
  G   = c(-105.0416159452, 0.9045834732, 9.0924508575, 8.9593522114, 8.5661099835),
  GL  = c(-111.3300139387, 0.8808081072, 9.0924508568, 8.9593676250, 8.5661957409),
  GLT = c(-110.0024198164, 0.6190925242, 9.0931260981, 8.9595732393, 8.5651544343)
)
# Hidden-unit summaries from the same reference fits, in layer order.
scale_precision <- rbind(
  G   = c(2.17460703, 2.17460703, 2.17460703, 3.16260552, 3.16260552),
  GL  = c(1.83936203, 2.50270043, 2.67869969, 2.76883108, 4.76525751),
  GLT = c(2.30804011, 3.38703879, 3.70459565, 3.49404682, 6.95845175)
)
geometric_prior_sd <- rbind(
  G   = c(0.76616554, 0.76616554, 0.76616554, 0.63910403, 0.63910403),
  GL  = c(0.96643271, 0.82851580, 0.80083531, 0.77003593, 0.58697018),
  GLT = c(0.98251826, 0.81489270, 0.78004070, 0.77618059, 0.55308122)
)
# Each pair is for layers 1 and 2. Layer scales use the log moment of the
# reference inverse-gamma factors, and tail shapes use direct integration
# of log(xi) against the reference variational density.
layer_scale <- rbind(G = c(0.71407744, 0.59565426),
                     GL = c(0.82371706, 0.78412675),
                     GLT = c(0.48557638, 0.46111219))
tail_shape <- c(1.32515933, 1.14431981)
set.seed(19)
before <- .Random.seed
for (prior in rownames(reference)) {
  fit <- gltbnn_fit(x, y, hidden = c(3L, 2L), prior = prior, control = control,
                    seed = 23L, geometry = "unit")
  means <- predict(fit, newx, draws = 32L, seed = 31L)
  estimates <- c(fit$engine$endpoint$value, fit$engine$state$noise$phi, means)
  stopifnot(length(estimates) == ncol(reference),
            all(abs(estimates - reference[prior, ]) < 1e-5))
  units <- gltbnn_shrinkage(fit)
  stopifnot(identical(units$layer, c(1L, 1L, 1L, 2L, 2L)),
            identical(units$unit, c(1L, 2L, 3L, 1L, 2L)),
            identical(units$group_size, c(3L, 3L, 3L, 4L, 4L)),
            all(abs(units$scale_precision - scale_precision[prior, ]) < 1e-5),
            all(abs(units$geometric_prior_sd - geometric_prior_sd[prior, ]) < 1e-5),
            all(abs(units$geometric_layer_scale - layer_scale[prior, units$layer]) < 1e-5))
  if (prior == "GLT") {
    stopifnot(all(abs(units$geometric_tail_shape - tail_shape[units$layer]) < 1e-5))
  } else {
    stopifnot(identical(units$geometric_tail_shape, rep(NA_real_, nrow(units))))
  }
}

latent <- predict(fit, newx, type = "mean_draws", draws = 32L, seed = 31L)
response <- predict(fit, newx, type = "response_draws", draws = 32L, seed = 31L)
stopifnot(identical(dim(latent), c(nrow(newx), 32L)),
          identical(dim(response), dim(latent)), all(is.finite(response)),
          isTRUE(all.equal(as.numeric(rowMeans(latent)), as.numeric(means))),
          any(response != latent), identical(.Random.seed, before))
