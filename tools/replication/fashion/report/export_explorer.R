# Export a compact demonstration after independent full-test scoring.
# Rscript --vanilla report/export_explorer.R data/data.rds results summary/scores.csv output
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) stop("Supply data.rds, the results directory, scores.csv and the output directory.")
data <- readRDS(args[1L])
scores <- read.csv(args[3L], stringsAsFactors = FALSE)
methods <- c("G", "GL", "GLT", "Unpenalized", "Dropout", "L1")
stopifnot(nrow(scores) == 36L, !anyDuplicated(scores[c("depth", "width", "method")]),
          setequal(scores$depth, c(2L, 5L, 10L)), setequal(scores$width, c(10L, 25L)),
          setequal(scores$method, methods), all(scores$n_train == 12000L), all(scores$n_test == 2000L),
          all(scores$correct / scores$n_test == scores$accuracy),
          identical(dim(data$Xtest), c(2000L, 784L)), identical(dim(data$Xtrain), c(12000L, 784L)),
          identical(tabulate(data$ytest + 1L), c(1000L, 1000L)), all(diff(data$test_index) > 0L))
rows <- c(head(which(data$ytest == 0L), 5L), head(which(data$ytest == 1L), 5L))
images <- lapply(rows, function(i) {
  pixels <- gsub("[[:space:]]", "", jsonlite::base64_enc(data$Xtest[i, ]))
  stopifnot(identical(jsonlite::base64_dec(pixels), data$Xtest[i, ]))
  list(id = data$test_index[i], truth = data$ytest[i], pixels = pixels)
})
moments <- vapply(seq_len(ncol(data$Xtrain)), function(j) {
  x <- as.integer(data$Xtrain[, j])
  c(mean(x), sd(x))
}, numeric(2L))
fits <- lapply(seq_len(nrow(scores)), function(i) {
  row <- scores[i, ]
  file <- with(row, sprintf("%s_d%d_w%d.rds", method, depth, width))
  fit <- readRDS(file.path(args[2L], file))
  stopifnot(fit$depth == row$depth, fit$width == row$width, identical(fit$method, row$method),
            identical(fit$test_id, data$test_index), identical(fit$truth, data$ytest),
            identical(colnames(fit$prob), c("T-shirt/top", "pullover")),
            abs(fit$accuracy - row$accuracy) < 1e-14, abs(fit$log_loss - row$log_loss) < 1e-12,
            max(abs(fit$preprocess$x_center * 255 - moments[1, ])) < 1e-10,
            max(abs(fit$preprocess$x_scale * 255 - moments[2, ])) < 1e-10)
  list(depth = row$depth, width = row$width, method = row$method, correct = row$correct,
       accuracy = row$accuracy, log_loss = row$log_loss, probabilities = unname(fit$prob[rows, , drop = FALSE]))
})
payload <- list(images = images, trainingMean = unname(moments[1, ]), trainingSD = unname(moments[2, ]), fits = fits)
json <- jsonlite::toJSON(payload, auto_unbox = TRUE, digits = 16, matrix = "rowmajor")
decoded <- jsonlite::fromJSON(json, simplifyVector = FALSE)
stopifnot(length(decoded$images) == 10L, length(decoded$fits) == 36L)
for (i in seq_along(decoded$fits)) {
  probabilities <- do.call(rbind, lapply(decoded$fits[[i]]$probabilities, unlist))
  stopifnot(max(abs(probabilities - fits[[i]]$probabilities)) < 1e-14)
}
dir.create(args[4L], recursive = TRUE, showWarnings = FALSE)
writeLines(paste0("window.BINARY_IMAGE_STUDY = ", json, ";"), file.path(args[4L], "data.js"), useBytes = TRUE)
stopifnot(file.copy(args[3L], file.path(args[4L], "scores.csv"), overwrite = TRUE))
