# Rebuild the public CIFAR-10 cat/dog split from its original binary release.
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
destination <- dirname(normalizePath(script))
archive_name <- "cifar-10-binary.tar.gz"
urls <- c("https://storage.googleapis.com/mirror.tensorflow.org/www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
          "https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz")
expected_md5 <- "c32a1d4ab5d03f1284b67883e8d87530"

read_batch <- function(path, offset) {
  stopifnot(file.size(path) == 3073L * 10000L)
  stream <- file(path, "rb")
  on.exit(close(stream))
  # Each original record begins with its zero-based CIFAR label, followed
  # by 1,024 red, 1,024 green and 1,024 blue bytes in row-major order.
  records <- readBin(stream, "raw", 3073L * 10000L)
  dim(records) <- c(3073L, 10000L)
  labels <- as.integer(records[1L, ])
  stopifnot(all(labels %in% 0:9))
  selected <- which(labels %in% c(3L, 5L))
  list(x = t(records[-1L, selected, drop = FALSE]),
       y = as.integer(labels[selected] == 5L), id = offset + selected)
}

prepare <- function() {
  cache <- Sys.getenv("GLTBNN_DATA_CACHE", tools::R_user_dir("gltbnn", "cache"))
  dir.create(cache, recursive = TRUE, showWarnings = FALSE)
  archive <- file.path(cache, archive_name)
  if (!file.exists(archive)) {
    tmp <- tempfile(tmpdir = cache)
    on.exit(unlink(tmp), add = TRUE)
    options(timeout = max(600, getOption("timeout")))
    for (url in urls) {
      status <- try(download.file(url, tmp, mode = "wb"), silent = TRUE)
      if (identical(status, 0L) && identical(unname(tools::md5sum(tmp)), expected_md5)) break
    }
    stopifnot(identical(unname(tools::md5sum(tmp)), expected_md5))
    if (!file.rename(tmp, archive) && !file.exists(archive))
      stop("Could not save the verified download.")
  }
  stopifnot(identical(unname(tools::md5sum(archive)), expected_md5))
  unpack <- tempfile()
  dir.create(unpack)
  on.exit(unlink(unpack, recursive = TRUE), add = TRUE)
  batch_files <- sprintf("data_batch_%d.bin", 1:5)
  subdir <- "cifar-10-batches-bin"
  members <- file.path(subdir, c(batch_files, "test_batch.bin", "batches.meta.txt"))
  utils::untar(archive, files = members, exdir = unpack)
  batch_dir <- file.path(unpack, subdir)
  labels <- readLines(file.path(batch_dir, "batches.meta.txt"))
  if (length(labels) == 11L && labels[11L] == "") labels <- head(labels, 10L)
  stopifnot(identical(labels, c("airplane", "automobile", "bird", "cat", "deer",
                                "dog", "frog", "horse", "ship", "truck")))
  batches <- lapply(seq_along(batch_files), function(b) {
    read_batch(file.path(batch_dir, batch_files[b]), (b - 1L) * 10000L)
  })
  test <- read_batch(file.path(batch_dir, "test_batch.bin"), 0L)
  data <- list(Xtrain = do.call(rbind, lapply(batches, "[[", "x")),
               ytrain = unlist(lapply(batches, "[[", "y"), use.names = FALSE),
               train_index = unlist(lapply(batches, "[[", "id"), use.names = FALSE),
               Xtest = test$x, ytest = test$y, test_index = test$id)
  stopifnot(is.raw(data$Xtrain), is.raw(data$Xtest),
            identical(dim(data$Xtrain), c(10000L, 3072L)),
            identical(dim(data$Xtest), c(2000L, 3072L)),
            identical(tabulate(data$ytrain + 1L), c(5000L, 5000L)),
            identical(tabulate(data$ytest + 1L), c(1000L, 1000L)),
            all(diff(data$train_index) > 0L), all(diff(data$test_index) > 0L))
  varying <- vapply(seq_len(ncol(data$Xtrain)), function(j) {
    x <- data$Xtrain[, j]
    any(x != x[1L])
  }, logical(1))
  stopifnot(all(varying))
  out <- file.path(destination, "data.rds")
  tmp_out <- tempfile(tmpdir = destination)
  on.exit(unlink(tmp_out), add = TRUE)
  saveRDS(data, tmp_out, version = 3L)
  if (file.exists(out)) {
    if (identical(readRDS(out), data)) return(invisible(NULL))
    stop("The current data.rds differs from the official cat/dog split.")
  }
  if (!file.rename(tmp_out, out)) stop("Could not install the verified split.")
}
prepare()
message("Verified 10,000 training and 2,000 test images with 3,072 native RGB values.")
