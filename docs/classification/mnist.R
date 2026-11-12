# Run Rscript mnist.R with gltbnn installed. Add --write-site to regenerate
# mnist.js and digits.png in the current directory (requires jsonlite).
# MNIST by Y. LeCun, C. Cortes and C. J. C. Burges. Digit images CC BY-SA 3.0.
# Originals and checksums: https://github.com/cvdfoundation/mnist
library(gltbnn)

cache <- Sys.getenv("GLTBNN_DATA_CACHE", tools::R_user_dir("gltbnn", "cache"))
dir.create(cache, recursive = TRUE, showWarnings = FALSE)
files <- c("train-images-idx3-ubyte.gz" = "f68b3c2dcbeaaa9fbdd348bbdeb94873",
           "train-labels-idx1-ubyte.gz" = "d53e105ee54ea40749a09fcbcd1e9432",
           "t10k-images-idx3-ubyte.gz" = "9fb629c4189551a2d022fa330f9573f3",
           "t10k-labels-idx1-ubyte.gz" = "ec29112dd5afa0611ce80d1b7f02629c")
for (name in names(files)) {
  path <- file.path(cache, name)
  if (!file.exists(path)) {
    download.file(paste0("https://storage.googleapis.com/cvdf-datasets/mnist/", name),
                  path, mode = "wb", quiet = TRUE)
  }
  stopifnot(unname(tools::md5sum(path)) == files[[name]])
}

read_split <- function(split, per_class, seed) {
  labels <- gzfile(file.path(cache, paste0(split, "-labels-idx1-ubyte.gz")), "rb")
  on.exit(close(labels))
  header <- readBin(labels, "integer", 2L, size = 4L, endian = "big")
  stopifnot(header[1L] == 2049L)
  y <- as.integer(readBin(labels, "raw", header[2L]))
  stopifnot(length(y) == header[2L])
  set.seed(seed)
  indices <- sort(unlist(lapply(0:9, function(k) sample(which(y == k), per_class))))
  images <- gzfile(file.path(cache, paste0(split, "-images-idx3-ubyte.gz")), "rb")
  on.exit(close(images), add = TRUE)
  dims <- readBin(images, "integer", 4L, size = 4L, endian = "big")
  stopifnot(identical(dims, c(2051L, header[2L], 28L, 28L)))
  # Keep the byte buffer raw until after subsetting to limit memory use.
  buffer <- readBin(images, "raw", dims[2L] * 784L)
  dim(buffer) <- c(784L, dims[2L])
  pixels <- t(buffer[, indices, drop = FALSE])
  list(index = indices, pixels = pixels, y = factor(y[indices], levels = 0:9))
}

features <- function(pixels) {
  # Number the pixels left to right in each row. Average disjoint 2 x 2
  # blocks. The package estimates centering/scaling on the training set.
  top_left <- as.vector(t(outer(seq(0L, 26L, 2L) * 28L, seq(1L, 27L, 2L), "+")))
  x <- matrix(as.numeric(pixels), nrow = nrow(pixels))
  (x[, top_left] + x[, top_left + 1L] + x[, top_left + 28L] + x[, top_left + 29L]) /
    (4 * 255)
}

train <- read_split("train", per_class = 100L, seed = 701L)
test <- read_split("t10k", per_class = 30L, seed = 702L)
fit <- gltbnn_fit(features(train$pixels), train$y, outcome = "categorical",
                  prior = "GLT", hidden = 20L, seed = 703L,
                  control = list(init_steps = 180L, outer = 12L, weight_steps = 3L,
                                 monitor_draws = 4L, endpoint_draws = 8L))
prob <- predict(fit, features(test$pixels), type = "prob", draws = 128L, seed = 704L)
truth <- as.integer(as.character(test$y))
predicted <- max.col(prob, ties.method = "first") - 1L
log_loss <- -mean(log(pmax(prob[cbind(seq_along(truth), truth + 1L)], .Machine$double.xmin)))
stopifnot(identical(colnames(prob), levels(train$y)), all(is.finite(prob)),
          max(abs(rowSums(prob) - 1)) < 1e-10)
print(table(Observed = test$y, Predicted = factor(predicted, levels = 0:9)))
print(c(accuracy = mean(truth == predicted), log_loss = log_loss))

if ("--write-site" %in% commandArgs(trailingOnly = TRUE)) {
  summary <- list(nTrain = nrow(train$pixels), sourceIndex = test$index, truth = truth,
                  predicted = predicted, probabilities = unname(prob), logLoss = log_loss)
  serialized <- jsonlite::toJSON(summary, digits = 12, auto_unbox = TRUE)
  writeBin(charToRaw(paste0("const mnist = ", serialized, ";\n")), "mnist.js")
  # Store the original 28 x 28 grayscale pixels in a 10-column contact sheet.
  cols <- 10L
  rows <- ceiling(length(truth) / cols)
  sheet <- matrix(0, nrow = 28L * rows, ncol = 28L * cols)
  for (i in seq_along(truth)) {
    ypos <- ((i - 1L) %/% cols) * 28L + seq_len(28L)
    xpos <- ((i - 1L) %% cols) * 28L + seq_len(28L)
    sheet[ypos, xpos] <- matrix(as.integer(test$pixels[i, ]), 28L, byrow = TRUE) / 255
  }
  grDevices::png("digits.png", width = ncol(sheet), height = nrow(sheet), bg = "black")
  grid::grid.raster(grDevices::as.raster(sheet), width = grid::unit(1, "npc"),
                    height = grid::unit(1, "npc"), interpolate = FALSE)
  grDevices::dev.off()
}
