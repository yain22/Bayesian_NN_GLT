# From the package source root, with R and jsonlite:
# Rscript --vanilla tools/generate-regularization-example.R
# Rscript --vanilla tools/generate-regularization-example.R --check
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 0L || identical(args, "--check"))
if (!file.exists("R/gltbnn.R")) stop("Run this script from the gltbnn source root.")
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Install jsonlite to write or check the example.")
pkg <- new.env(parent = globalenv())
for (f in list.files("R", pattern = "[.]R$", full.names = TRUE)) sys.source(f, pkg)
study <- new.env(parent = globalenv())
sys.source("tools/replication/c2-additive.R", study)
RNGkind("Mersenne-Twister", "Inversion", "Rejection")

data <- datasets::trees
holdout <- seq(5L, nrow(data), by = 5L)
train <- setdiff(seq_len(nrow(data)), holdout)
test_row <- 25L
stopifnot(test_row %in% holdout)
features <- c("Girth", "Height")
raw_x <- as.matrix(data[train, features])
raw_y <- data$Volume[train]
prep <- list(x_center = colMeans(raw_x), x_scale = apply(raw_x, 2L, sd),
             y_center = mean(raw_y), y_scale = sd(raw_y))
x <- pkg$.gltbnn_scale_x(raw_x, prep)
y <- (raw_y - prep$y_center) / prep$y_scale
raw_test <- as.matrix(data[test_row, features])
x_test <- pkg$.gltbnn_scale_x(raw_test, prep)
to_volume <- function(f) unname(f * prep$y_scale + prep$y_center)
same <- function(a, b) {
  stopifnot(isTRUE(all.equal(as.numeric(a), as.numeric(b), tolerance = 1e-10)))
}

hidden <- c(3L, 2L)
warm_seed <- 7L
fit_seed <- 32L
prediction_seed <- 33L
draws <- 256L
controls <- pkg$.gltbnn_control(list())
stopifnot(controls$init_steps == 600L, controls$outer * controls$weight_steps == 600L)
drop_rate <- .2
penalty <- .001

# This is the study's full-batch unpenalized warm start. Pass the very same
# matrices to every fit. The network paths are active for this initialization.
start <- study$train_network(x, y, hidden, warm_seed)
first_weight <- function(weights) unname(weights[[1L]][1L, 1L])
start_weight <- first_weight(start)

# Replay the first full training pass under dropout. Select one training tree
# whose first hidden unit is active and actually masked on this pass.
set.seed(fit_seed)
masked <- study$forward_pass(x, start, dropout = drop_rate)
unmasked <- study$forward_pass(x, start, dropout = 0)
unit <- 1L
activation <- pmax(masked$preactivation[[1L]][, unit], 0)
multiplier <- masked$masks[[1L]][, unit]
example_rows <- c(which(activation > 0 & multiplier == 0)[1L],
                  which(activation > 0 & multiplier > 0)[1L])
stopifnot(!anyNA(example_rows))
training_pos <- example_rows[1L]
passed <- masked$inputs[[2L]][example_rows, unit]
same(passed, activation[example_rows] * multiplier[example_rows])
examples <- data.frame(tree_row = train[example_rows], hidden_activation = activation[example_rows],
                       multiplier = multiplier[example_rows], passed_activation = passed)

training_row <- function(weight, z, multiplier, passed_activation) {
  list(weight = unname(weight), preactivation = unname(z), activation = max(z, 0),
       multiplier = unname(multiplier), passed_activation = unname(passed_activation))
}

fit_bnn <- function(prior) {
  e <- pkg$.gltbnn_engine_fit(x, y, hidden, prior, controls, fit_seed, initial_weights = start)
  stopifnot(e$outer == 60L, nrow(e$weight_steps) == 600L)
  fit <- structure(list(prior = prior, outcome = "continuous", geometry = "coefficient",
                        hidden = hidden, p = ncol(x), preprocess = prep,
                        predictor_names = features, engine = e), class = "gltbnn_fit")
  scales <- pkg$gltbnn_shrinkage(fit)
  scale_after <- with(scales, scale_precision[layer == 1L & unit == 1L & coordinate == 1L])
  stopifnot(length(scale_after) == 1L)
  initial_terms <- lapply(e$config$M, function(count) {
    group <- pkg$.bnn_init_groups(count, 1L, prior, "coefficient")
    pkg$.bnn_prior_terms(group, count, prior)
  })
  scale_before <- initial_terms[[1L]]$precision[1L]
  phi_before <- e$trace$noise_precision[1L]
  phi_after <- e$state$noise$phi

  # The engine seeds its VI stream, draws the monitoring bank once, and then
  # draws the four perturbations used by the first Adam gradient and line search.
  initial <- list(mu = start, logsd = lapply(start, function(w) {
    matrix(log(.01), nrow(w), ncol(w))
  }))
  set.seed(fit_seed + 1L)
  invisible(pkg$.bnn_antithetic_draws(initial, controls$monitor_draws))
  eps <- pkg$.bnn_antithetic_draws(initial, controls$weight_draws)
  rates <- c(lapply(initial_terms, function(pr) phi_before * pr$precision), list(1))
  value <- pkg$.bnn_weight_objective(initial, x, y, phi_before, rates, eps)
  same(value, e$weight_steps$sample_objective[1L])
  training_pass <- lapply(eps, function(z) {
    w <- pkg$.bnn_draw_weights(initial, z)
    pass <- pkg$.bnn_forward(x[training_pos, , drop = FALSE], w)
    pre <- pass$pre[[1L]][1L, unit]
    row <- training_row(first_weight(w), pre, 1, pass$features[[2L]][1L, unit])
    row$epsilon <- first_weight(z)
    row
  })

  # Keep each featured coefficient with the output of its entire sampled network.
  # Verify every volume against predict(type="mean_draws") in the same draw order.
  pairs <- pkg$.gltbnn_with_seed(prediction_seed, replicate(draws, {
    w <- pkg$.bnn_draw_weights(e$parameters, pkg$.bnn_normal_draw(e$parameters))
    c(weight = first_weight(w), volume = to_volume(pkg$.bnn_forward(x_test, w)$y))
  }))
  reported <- pkg$predict.gltbnn_fit(fit, raw_test, type = "mean_draws", draws = draws,
                                      seed = prediction_seed)
  same(pairs["volume", ], reported)
  mean_volume <- mean(pairs["volume", ])
  same(mean_volume, pkg$predict.gltbnn_fit(fit, raw_test, type = "mean", draws = draws,
                                           seed = prediction_seed))
  bank <- as.data.frame(t(pairs))
  list(weight_before = start_weight, weight_after = first_weight(e$parameters$mu),
       sd_before = .01, sd_after = exp(first_weight(e$parameters$logsd)),
       scale_precision_before = scale_before, scale_precision_after = scale_after,
       noise_precision_before = phi_before, noise_precision_after = phi_after,
       effective_precision_before = phi_before * scale_before,
       effective_precision_after = phi_after * scale_after, training_pass = training_pass,
       prediction = list(mean = mean_volume, draws = bank))
}

fit_fnn <- function(name) {
  lambda <- if (name == "L1") penalty else 0
  drop <- if (name == "Dropout") drop_rate else 0
  w <- study$train_network(x, y, hidden, fit_seed, weights = start, dropout = drop, l1 = lambda)
  test_pass <- study$forward_pass(x_test, w, dropout = 0)
  activation <- as.numeric(pmax(test_pass$preactivation[[1L]], 0))
  stopifnot(all(test_pass$masks[[1L]] == 1))
  pass <- if (name == "Dropout") masked else unmasked
  first_training <- training_row(start_weight, pass$preactivation[[1L]][training_pos, unit],
                                 pass$masks[[1L]][training_pos, unit],
                                 pass$inputs[[2L]][training_pos, unit])
  result <- list(weight_before = start_weight, weight_after = first_weight(w),
                 first_layer_activations = activation, training_pass = list(first_training),
                 prediction = list(mean = to_volume(test_pass$y), draws = list()))
  if (name == "L1") {
    result$penalty_subgradient_before <- lambda * sign(start_weight)
    result$penalty_subgradient_after <- lambda * sign(result$weight_after)
  }
  result
}

bnn <- c("G", "GL", "GLT")
fnn <- c("FNN", "Dropout", "L1")
methods <- c(setNames(lapply(bnn, fit_bnn), bnn), setNames(lapply(fnn, fit_fnn), fnn))

selected_unit <- which.max(methods$Dropout$first_layer_activations)
stopifnot(methods$Dropout$first_layer_activations[selected_unit] > 0)

fixture <- list(
  dataset = "datasets::trees", n_train = length(train), held_out_rows = holdout,
  response_scaling = list(mean = prep$y_center, sd = prep$y_scale),
  coefficient = list(layer = 1L, unit = 1L, coordinate = 1L, predictor = "Girth"),
  tree = list(row = test_row, diameter = data$Girth[test_row], height = data$Height[test_row],
              observed_volume = data$Volume[test_row], standardized_input = as.numeric(x_test)),
  training_tree = list(row = train[training_pos], diameter = raw_x[training_pos, 1L],
                       height = raw_x[training_pos, 2L], observed_volume = raw_y[training_pos],
                       standardized_input = as.numeric(x[training_pos, ])),
  control = list(hidden = hidden, warm_seed = warm_seed, fit_seed = fit_seed,
                 prediction_seed = prediction_seed, init_steps = controls$init_steps,
                 fnn_steps = 600L, bnn_cycles = controls$outer, weight_steps = controls$weight_steps,
                 learning_rate = controls$lr, weight_draws = controls$weight_draws,
                 noise_draws = controls$noise_draws, prediction_draws = draws, display_draws = draws),
  dropout = list(probability = drop_rate, retained_multiplier = 1 / (1 - drop_rate),
                 training_unit = unit, examples = examples, selected_unit = selected_unit),
  l1 = list(coefficient = penalty, includes_biases = TRUE),
  methods = methods
)
target <- "docs/data/regularization-example.json"
json <- jsonlite::toJSON(fixture, pretty = TRUE, auto_unbox = TRUE, digits = 12)
if (identical(args, "--check")) {
  reference <- jsonlite::fromJSON(target, simplifyVector = FALSE)
  generated <- jsonlite::fromJSON(json, simplifyVector = FALSE)
  stopifnot(isTRUE(all.equal(reference, generated, tolerance = 1e-9)))
  message("Six refits reproduce the common-start comparison and paired draws.")
} else {
  writeLines(json, target, useBytes = TRUE)
}
