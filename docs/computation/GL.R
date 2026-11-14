library(gltbnn)
x <- datasets::trees[c("Girth", "Height")]
y <- datasets::trees$Volume
test <- seq(5L, nrow(x), by = 5L)
fit <- gltbnn_fit(x[-test, ], y[-test], hidden = c(3L, 2L),
                  prior = "GL", seed = 31L)
scales <- gltbnn_shrinkage(fit)
subset(scales, layer == 1 & unit == 1,
       select = c(coordinate, is_bias, scale_precision, geometric_prior_sd))
predict(fit, x[test, ], type = "mean", draws = 256L, seed = 32L)
