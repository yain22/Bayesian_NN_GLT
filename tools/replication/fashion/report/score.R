# Rscript score.R data.rds results/ summary_dir/
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Supply a split file, a results directory and an output directory.")
data <- readRDS(args[1L])
root <- dirname(dirname(normalizePath(args[1L])))
protocol <- readRDS(file.path(root, "protocol.rds"))
run_id <- digest::digest(protocol, algo = "md5", serializeVersion = 2L)
sources <- c("compute/run_full.R", "compute/full_fit.R", "compute/dropout.R", "compute/engine-fit-batches.R",
             sort(paste0("package/R/", list.files(file.path(root, "package/R"), pattern = "[.]R$"))))
stopifnot(setequal(names(protocol$hashes), c("data/data.rds", sources)),
          identical(unname(tools::md5sum(file.path(root, sources))), unname(protocol$hashes[sources])),
          identical(protocol$rng, RNGkind()), identical(protocol$study, "Fashion-MNIST T-shirt/top and pullover"))
# Authenticate the exact native row-major pixels, labels and IDs independently of RDS compression.
content_digests <- c(
  Xtrain = "ad1555e8941f5e538a9c28e94ed983c92df40fb3ca7358ce35d4af9cc544691e", Xtest = "e2699ac4cb811a2f624d46ed5923eecc90568e1a196e59b922cd49955f2cd315",
  ytrain = "0326f3d9aa4f5e48f04142e831fd87bf77082c2903fc458ada62c76ba9a0426d", ytest = "5aebafc981cb7caaa6d6662911df38fe88659af79ea73e4df36ca7febda77d08",
  train_index = "a071d92019682fb416e35b4f881b4193f9ed696e2eac99615cec94ac32361f2b", test_index = "e917e58ae15892ab35dbdbdeeabccf551a379dc8ca649e2962dbd7116e927d92")
content_sha <- function(name) {
  x <- data[[name]]
  bytes <- if (is.matrix(x)) {
    as.vector(t(x))
  } else if (endsWith(name, "_index")) {
    writeBin(as.integer(x), raw(), size = 4L, endian = "little")
  } else as.raw(x)
  digest::digest(bytes, algo = "sha256", serialize = FALSE)
}
stopifnot(identical(content_digests, vapply(names(content_digests), content_sha, character(1L))))
classes <- c("T-shirt/top", "pullover")
truth <- data$ytest
training_labels <- data$ytrain
ntest <- length(truth)
ntrain <- length(training_labels)
stopifnot(ntest == 2000L, ntrain == 12000L, identical(dim(data$Xtest), c(ntest, 784L)),
          identical(dim(data$Xtrain), c(ntrain, 784L)), is.integer(truth), is.integer(training_labels),
          is.raw(data$Xtrain), is.raw(data$Xtest), all(truth %in% 0:1), all(training_labels %in% 0:1),
          identical(tabulate(truth + 1L, 2L), c(1000L, 1000L)),
          identical(tabulate(training_labels + 1L, 2L), c(6000L, 6000L)),
          all(diff(data$train_index) > 0L), all(diff(data$test_index) > 0L))
moments <- vapply(seq_len(784L), function(j) {
  column <- as.integer(data$Xtrain[, j])
  c(mean(column), sd(column))
}, numeric(2))
train_center <- moments[1L, ]
train_sd <- moments[2L, ]
keep <- which(train_sd > 0)
stopifnot(length(keep) == 784L)
xtest <- matrix(as.integer(data$Xtest[, keep, drop = FALSE]), nrow = ntest)
xtest <- sweep(xtest, 2L, train_center[keep], "-")
xtest <- sweep(xtest, 2L, train_sd[keep], "/")
methods <- c("G", "GL", "GLT", "Unpenalized", "Dropout", "L1")
grid <- expand.grid(method = methods, width = c(10L, 25L), depth = c(2L, 5L, 10L),
                    stringsAsFactors = FALSE)
keys <- with(grid, sprintf("%s_d%d_w%d.rds", method, depth, width))
stopifnot(setequal(keys, list.files(args[2L], pattern = "[.]rds$")))

# Reconstruct both Bernoulli log probabilities from the saved pullover logit.
# Working on the log scale also recovers the loss when a probability underflows.
log_probs <- function(x, weights, width, depth) {
  stopifnot(identical(vapply(weights, nrow, integer(1)), c(rep(width, depth), 1L)),
            identical(vapply(weights, ncol, integer(1)), c(length(keep), rep(width, depth)) + 1L))
  h <- x
  for (l in seq_along(weights)) {
    h <- cbind(h, 1) %*% t(weights[[l]])
    if (l < length(weights)) h <- pmax(h, 0)
  }
  eta <- as.vector(h)
  cbind("T-shirt/top" = plogis(eta, lower.tail = FALSE, log.p = TRUE),
        pullover = plogis(eta, log.p = TRUE))
}

# Independently replay the two shuffled schedules once per architecture. In
# each phase, all 12,000 records appear exactly once in each of 50 complete passes.
schedule <- function(seed) {
  set.seed(seed)
  indices <- list()
  for (epoch in seq_len(50L)) {
    permutation <- sample.int(ntrain)
    starts <- seq.int(1L, ntrain, by = 1024L)
    batches <- lapply(starts, function(j) permutation[j:min(ntrain, j + 1023L)])
    stopifnot(identical(lengths(batches), c(rep(1024L, 11L), 736L)),
              identical(sort(unlist(batches)), seq_len(ntrain)))
    indices <- c(indices, batches)
  }
  stopifnot(length(indices) == 600L)
  visits <- tabulate(unlist(indices), nbins = ntrain)
  stopifnot(all(visits == 50L))
  digest::digest(indices, algo = "md5", serializeVersion = 2L)
}
schedule_cache <- new.env(parent = emptyenv())
check_schedule <- function(fit) {
  s <- fit$schedule
  stopifnot(s$warmup_seed == fit$seed + 40000L, s$continuation_seed == fit$seed + 50000L,
            s$batch_size == 1024L, s$steps == 600L,
            isTRUE(grepl("^[a-f0-9]{32}$", s$initialization_md5)))
  key <- as.character(fit$seed)
  if (!exists(key, schedule_cache, inherits = FALSE)) {
    init <- readRDS(file.path(root, sprintf("initializers/d%d_w%d.rds", fit$depth, fit$width)))
    init_hash <- digest::digest(init$weights, algo = "md5", serializeVersion = 2L)
    warmup_hash <- schedule(s$warmup_seed)
    stopifnot(identical(init$run_id, run_id), identical(init$seed, fit$seed),
              identical(init_hash, init$initialization_md5), identical(warmup_hash, init$warmup_md5))
    schedule_cache[[key]] <- c(warmup_hash, schedule(s$continuation_seed), init_hash)
  }
  stopifnot(identical(c(s$warmup_md5, s$continuation_md5, s$initialization_md5), schedule_cache[[key]]))
}

# Rebuild posterior predictive probabilities from the fitted variational
# parameters for the first five held-out records of each class. The remaining
# records are validated from their archived probability vectors and log scores.
check_draws <- function(fit, width, depth) {
  idx <- c(head(which(truth == 0L), 5L), head(which(truth == 1L), 5L))
  p <- fit$parameters
  scales <- lapply(p$logsd, exp)
  stopifnot(identical(lapply(p$mu, dim), lapply(scales, dim)))
  estimates <- matrix(0, length(idx), 2L)
  logged <- matrix(0, length(idx), fit$predict_draws)
  set.seed(fit$predict_seed)
  for (b in seq_len(fit$predict_draws)) {
    w <- Map(function(m, s) m + s * rnorm(length(m)), p$mu, scales)
    lp <- log_probs(xtest[idx, , drop = FALSE], w, width, depth)
    estimates <- estimates + exp(lp)
    logged[, b] <- lp[cbind(seq_along(idx), truth[idx] + 1L)]
  }
  largest <- apply(logged, 1L, max)
  truth_log <- largest + log(rowMeans(exp(logged - largest)))
  stopifnot(max(abs(estimates / fit$predict_draws - fit$prob[idx, ])) < 1e-9,
            max(abs(truth_log - fit$true_logprob[idx])) < 1e-8)
}

check_fit <- function(i) {
  fit <- readRDS(file.path(args[2L], keys[i]))
  with(grid[i, ], {
    stopifnot(identical(fit$run_id, run_id), identical(fit$method, method),
              identical(fit$depth, depth), identical(fit$width, width),
              identical(fit$test_id, data$test_index), identical(fit$truth, truth),
              identical(fit$seed, 241000L + 100L * depth + width))
    stopifnot(identical(fit$train_id, data$train_index), identical(fit$levels, classes),
              identical(fit$reference, "T-shirt/top"), identical(fit$positive, "pullover"), identical(fit$retained_inputs, keep),
              fit$n_train == ntrain, identical(fit$preprocess$x_keep, keep),
              length(fit$preprocess$x_center) == 784L, length(fit$preprocess$x_scale) == 784L,
              max(abs(fit$preprocess$x_center - train_center / 255)) < 1e-12,
              max(abs(fit$preprocess$x_scale - train_sd / 255)) < 1e-12)
    check_schedule(fit)
    prob <- fit$prob
    stopifnot(is.matrix(prob), identical(dim(prob), c(ntest, 2L)), identical(colnames(prob), classes),
              all(is.finite(prob)), all(prob >= 0 & prob <= 1 + 1e-12),
              max(abs(rowSums(prob) - 1)) < 1e-10)
    pred <- max.col(prob, ties.method = "first") - 1L
    observed <- cbind(seq_along(truth), truth + 1L)
    stopifnot(length(fit$true_logprob) == ntest, all(is.finite(fit$true_logprob)),
              all(fit$true_logprob <= 1e-12),
              max(abs(exp(fit$true_logprob) - prob[observed])) < 1e-10)
    representable <- prob[observed] >= .Machine$double.xmin
    stopifnot(max(abs(log(prob[observed][representable]) - fit$true_logprob[representable])) < 1e-9)
    if (method %in% c("G", "GL", "GLT")) {
      cfg <- fit$config
      stopifnot(identical(cfg$K, rep(width, depth)), cfg$n == ntrain, cfg$p == length(keep),
                cfg$L == depth, identical(cfg$geometry, "coefficient"), identical(cfg$prior, method),
                identical(cfg$outcome, "binary"), cfg$n_classes == 2L,
                identical(fit$trace$outer, 0:60), all(is.na(fit$trace$elbo)),
                identical(fit$weight_steps$step, 1:600), all(is.finite(fit$weight_steps$sample_objective)),
                all(is.finite(fit$weight_steps$sample_gain)), fit$control$weight_draws == 4L,
                fit$control$outer == 60L, fit$control$weight_steps == 10L,
                fit$control$init_steps == 600L, fit$predict_draws == 256L,
                fit$predict_seed == fit$seed + 10000L,
                sum(fit$trace$accepted + fit$trace$rejected) == 600L)
      check_draws(fit, width, depth)
    } else {
      stopifnot(fit$steps == 600L, fit$init_steps == 600L,
                fit$dropout == if (method == "Dropout") .2 else 0,
                fit$l1 == if (method == "L1") .001 else 0)
      lp <- log_probs(xtest, fit$weights, width, depth)
      stopifnot(max(abs(exp(lp) - prob)) < 1e-9, max(abs(lp[observed] - fit$true_logprob)) < 1e-8)
    }
    correct <- sum(pred == truth)
    log_loss <- -mean(fit$true_logprob)
    stopifnot(identical(fit$pred, pred), is.finite(log_loss), is.finite(fit$log_loss),
              abs(log_loss - fit$log_loss) < 1e-9 * max(1, log_loss),
              identical(as.numeric(fit$accuracy), correct / ntest))
    data.frame(depth, width, method, n_train = ntrain, n_test = ntest, correct,
               accuracy = correct / ntest, log_loss = log_loss)
  })
}
scores <- do.call(rbind, lapply(seq_len(nrow(grid)), check_fit))

panel_rows <- function(metric, percent = FALSE) {
  rows <- split(seq_len(nrow(scores)), rep(1:6, each = length(methods)))
  vapply(rows, function(row) {
    z <- scores[row, ]
    stopifnot(identical(z$method, methods))
    value <- z[[metric]]
    best <- if (percent) max(value) else min(value)
    values <- if (percent) sprintf("%.2f", 100 * value) else sprintf("%.3f", value)
    winners <- value == best
    values[winners] <- sprintf("\\textbf{%s}", values[winners])
    label <- sprintf("$%d\\times %d$", z$depth[1L], z$width[1L])
    paste(paste(c(label, values), collapse = " & "), "\\\\")
  }, "")
}
table_tex <- c(
  "\\begin{table}[htbp]", "\\centering\\small",
  "\\caption{T-shirt/top versus pullover classification on the 2,000 images of these classes in the official Fashion-MNIST test set. All fits use the 12,000 images of these classes in the official training set. Bold marks the highest accuracy or lowest mean log loss within each row, including ties at full precision.}\\label{tab:fashion-results}",
  "\\begin{tabular}{@{}lrrrrrr@{}}\\toprule",
  "& \\multicolumn{3}{c}{BNN} & \\multicolumn{3}{c}{FNN} \\\\",
  "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
  "Depth $\\times$ width & G & GL & GLT & Unpenalized & Dropout & $L_1$ \\\\",
  "\\midrule", "\\multicolumn{7}{l}{(a) Accuracy (\\%)} \\\\", panel_rows("accuracy", percent = TRUE),
  "\\midrule", "\\multicolumn{7}{l}{(b) Mean log loss} \\\\", panel_rows("log_loss"),
  "\\bottomrule\\end{tabular}", "\\end{table}")
dir.create(args[3L], recursive = TRUE, showWarnings = FALSE)
write.csv(scores, file.path(args[3L], "scores.csv"), row.names = FALSE)
writeLines(table_tex, file.path(args[3L], "v3_fashion_results.tex"))
cat("Validated 36 fits and 72,000 held-out predictions. Wrote the scores and two-panel table.\n")
