library(gltbnn)
x <- datasets::trees[c("Girth", "Height")]
y <- datasets::trees$Volume
test <- seq(5L, nrow(x), by = 5L)
fit <- gltbnn_fit(x[-test, ], y[-test], hidden = c(3L, 2L),
                  prior = "G", seed = 31L)
scales <- gltbnn_shrinkage(fit)
unique(scales[c("layer", "geometric_layer_scale", "scale_precision")])
predict(fit, x[test, ], type = "mean", draws = 256L, seed = 32L)
