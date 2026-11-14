# Run from any directory with Rscript data/prepare.R.
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
directory <- dirname(normalizePath(script))
cache <- Sys.getenv("GLTBNN_DATA_CACHE", tools::R_user_dir("gltbnn", "cache"))
dir.create(cache, recursive = TRUE, showWarnings = FALSE)

md5 <- c("train-images-idx3-ubyte.gz" = "f68b3c2dcbeaaa9fbdd348bbdeb94873",
         "train-labels-idx1-ubyte.gz" = "d53e105ee54ea40749a09fcbcd1e9432",
         "t10k-images-idx3-ubyte.gz" = "9fb629c4189551a2d022fa330f9573f3",
         "t10k-labels-idx1-ubyte.gz" = "ec29112dd5afa0611ce80d1b7f02629c")
for (filename in names(md5)) {
  path <- file.path(cache, filename)
  if (!file.exists(path)) {
    download.file(paste0("https://storage.googleapis.com/cvdf-datasets/mnist/", filename),
                  path, mode = "wb")
  }
  stopifnot(identical(unname(tools::md5sum(path)), md5[[filename]]))
}

read_idx <- function(filename, header, size) {
  connection <- gzfile(file.path(cache, filename), "rb")
  on.exit(close(connection))
  stopifnot(identical(readBin(connection, "integer", length(header), size = 4L,
                              endian = "big"), header))
  values <- readBin(connection, "raw", size)
  stopifnot(length(values) == size, length(readBin(connection, "raw", 1L)) == 0L)
  values
}

read_split <- function(prefix, n) {
  pixels <- read_idx(paste0(prefix, "-images-idx3-ubyte.gz"), c(2051L, n, 28L, 28L),
                     n * 784L)
  # A row of the returned matrix is one image in native IDX pixel order.
  dim(pixels) <- c(784L, n)
  labels <- read_idx(paste0(prefix, "-labels-idx1-ubyte.gz"), c(2049L, n), n)
  list(x = t(pixels), y = as.integer(labels))
}

train <- read_split("train", 60000L)
test <- read_split("t10k", 10000L)
stopifnot(identical(tabulate(train$y + 1L, 10L),
                    c(5923L, 6742L, 5958L, 6131L, 5842L, 5421L, 5918L, 6265L, 5851L, 5949L)),
          identical(tabulate(test$y + 1L, 10L),
                    c(980L, 1135L, 1032L, 1010L, 982L, 892L, 958L, 1028L, 974L, 1009L)))
varying <- vapply(seq_len(ncol(train$x)), function(j) {
  x <- train$x[, j]
  any(x != x[1L])
}, logical(1))
stopifnot(sum(varying) == 717L)

data <- list(Xtrain = train$x, ytrain = train$y, train_index = seq_along(train$y),
             Xtest = test$x, ytest = test$y, test_index = seq_along(test$y))
saveRDS(data, file.path(directory, "data.rds"), version = 3L)
message("Saved 60,000 training and 10,000 test images with all 784 native pixels.")
