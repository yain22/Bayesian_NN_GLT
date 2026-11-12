# Run Rscript wisconsin.R with gltbnn installed. Add --write-site to
# regenerate wisconsin.js in the current directory (requires jsonlite).
# Wolberg et al., Breast Cancer Wisconsin (Diagnostic), UCI, CC BY 4.0.
# https://doi.org/10.24432/C5DW2B
library(gltbnn)

cache <- Sys.getenv("GLTBNN_DATA_CACHE", tools::R_user_dir("gltbnn", "cache"))
dir.create(cache, recursive = TRUE, showWarnings = FALSE)
archive <- file.path(cache, "wdbc.zip")
url <- "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
if (!file.exists(archive)) download.file(url, archive, mode = "wb", quiet = TRUE)
stopifnot(unname(tools::md5sum(archive)) == "97da86fb8aa67b905c88ad6c499f6bd4")
data <- local({
  tmp <- tempfile("wdbc-")
  dir.create(tmp)
  on.exit(unlink(tmp, recursive = TRUE))
  member <- utils::unzip(archive, files = "wdbc.data", exdir = tmp)
  read.csv(member, header = FALSE, colClasses = "character")
})
stopifnot(nrow(data) == 569L, ncol(data) == 32L)

# Column 1 identifies the record. Columns 3:32 contain image measurements.
y <- factor(data[[2]], levels = c("B", "M"), labels = c("benign", "malignant"))
x <- as.data.frame(lapply(data[3:32], as.numeric))
set.seed(610)
holdout <- sort(unlist(lapply(split(seq_along(y), y), function(rows) {
  sample(rows, round(length(rows) / 5))
}), use.names = FALSE))

fit <- gltbnn_fit(x[-holdout, ], y[-holdout], outcome = "binary",
                  prior = "GLT", hidden = 8L, seed = 641L,
                  control = list(init_steps = 240L, outer = 15L, weight_steps = 4L,
                                 monitor_draws = 8L, endpoint_draws = 16L))
prob <- predict(fit, x[holdout, ], type = "prob", draws = 256L, seed = 652L)
truth <- y[holdout]
malignant <- truth == "malignant"
p <- prob[, "malignant"]
log_loss <- -mean(log(pmax(prob[cbind(seq_along(truth), as.integer(truth))],
                           .Machine$double.xmin)))
brier <- mean((p - malignant)^2)
thresholds <- (1:19) / 20
counts <- t(vapply(thresholds, function(t) {
  positive <- p >= t
  c(tp = sum(positive & malignant), fn = sum(!positive & malignant),
    fp = sum(positive & !malignant), tn = sum(!positive & !malignant))
}, numeric(4)))
stopifnot(identical(colnames(prob), levels(y)), all(is.finite(prob)),
          max(abs(rowSums(prob) - 1)) < 1e-10, all(rowSums(counts) == length(truth)))
print(table(Observed = truth, Predicted = factor(ifelse(p >= 0.5, "malignant", "benign"),
                                                levels = levels(y))))
print(c(log_loss = log_loss, Brier_score = brier))

if ("--write-site" %in% commandArgs(trailingOnly = TRUE)) {
  summary <- list(nTrain = nrow(x) - length(holdout), nTest = length(holdout),
                  logLoss = log_loss, brier = brier, counts = unname(counts),
                  thresholds = thresholds)
  serialized <- jsonlite::toJSON(summary, digits = 12, auto_unbox = TRUE)
  writeBin(charToRaw(paste0("const wisconsin = ", serialized, ";\n")), "wisconsin.js")
}
