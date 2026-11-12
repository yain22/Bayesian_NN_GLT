# Rebuild the binary Fashion-MNIST dataset from the original four IDX files.
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
destination <- dirname(normalizePath(script))
cache <- file.path(Sys.getenv("GLTBNN_DATA_CACHE", tools::R_user_dir("gltbnn", "cache")), "fashion-mnist")
dir.create(cache, recursive = TRUE, showWarnings = FALSE)
expected <- c("train-images-idx3-ubyte.gz" = "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
              "train-labels-idx1-ubyte.gz" = "25c81989df183df01b3e8a0aad5dffbe",
              "t10k-images-idx3-ubyte.gz" = "bef4ecab320f06d8554ea6380940ec79",
              "t10k-labels-idx1-ubyte.gz" = "bb300cfdad3c16e7a12a480ee83cd310")
base_url <- "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"

ensure_file <- function(name) {
  path <- file.path(cache, name)
  if (!file.exists(path)) {
    tmp <- tempfile(tmpdir = cache)
    on.exit(unlink(tmp))
    options(timeout = max(300, getOption("timeout")))
    download.file(paste0(base_url, name), tmp, mode = "wb")
    stopifnot(identical(unname(tools::md5sum(tmp)), expected[[name]]))
    if (!file.rename(tmp, path) && !file.exists(path)) stop("Could not save the verified file.")
  }
  stopifnot(identical(unname(tools::md5sum(path)), expected[[name]]))
}
invisible(lapply(names(expected), ensure_file))

read_idx <- function(name, header, size) {
  stream <- gzfile(file.path(cache, name), "rb")
  on.exit(close(stream))
  stopifnot(identical(readBin(stream, "integer", length(header), size = 4L, endian = "big"), header))
  contents <- readBin(stream, "raw", size)
  stopifnot(length(contents) == size, length(readBin(stream, "raw", 1L)) == 0L)
  contents
}

get_pair <- function(prefix, n) {
  labels <- as.integer(read_idx(paste0(prefix, "-labels-idx1-ubyte.gz"), c(2049L, n), n))
  stopifnot(identical(tabulate(labels + 1L, nbins = 10L), rep(n %/% 10L, 10L)))
  id <- which(labels %in% c(0L, 2L))
  pixels <- read_idx(paste0(prefix, "-images-idx3-ubyte.gz"), c(2051L, n, 28L, 28L), 784L * n)
  dim(pixels) <- c(784L, n)
  list(x = t(pixels[, id, drop = FALSE]), y = as.integer(labels[id] == 2L), id = id)
}

save_split <- function() {
  train <- get_pair("train", 60000L)
  test <- get_pair("t10k", 10000L)
  data <- list(Xtrain = train$x, ytrain = train$y, train_index = train$id,
               Xtest = test$x, ytest = test$y, test_index = test$id)
  stopifnot(identical(dim(data$Xtrain), c(12000L, 784L)), identical(dim(data$Xtest), c(2000L, 784L)),
            identical(tabulate(data$ytrain+1L), c(6000L, 6000L)), identical(tabulate(data$ytest+1L), c(1000L, 1000L)),
            all(diff(data$train_index) > 0L), all(diff(data$test_index) > 0L))
  out <- file.path(destination, "data.rds")
  if (file.exists(out)) {
    if (identical(readRDS(out), data)) return(invisible(NULL))
    stop("The existing study split differs from the verified original records.")
  }
  tmp <- tempfile(tmpdir = destination)
  on.exit(unlink(tmp))
  saveRDS(data, tmp, version = 3L)
  if (!file.rename(tmp, out)) stop("Could not install the verified split.")
}
save_split()
message("Verified 12,000 training and 2,000 held-out T-shirt/top and pullover images.")
