# Run from the folder containing this file, e.g. Rscript --vanilla run.R 12
main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  workers <- if (length(args)) as.integer(args[1]) else 1L
  stopifnot(is.finite(workers), workers >= 1L)
  Sys.setenv(OPENBLAS_NUM_THREADS = 1L, OMP_NUM_THREADS = 1L, MKL_NUM_THREADS = 1L)
  source("networks.R")
  input <- file.path("..", "data", "analysis.csv")
  data <- read.csv(input)
  stopifnot(nrow(data) == 9915L, !anyNA(data), identical(data$row_id, seq_len(nrow(data))),
            all(table(data$fold) == 1983L), sum(data$e401) == 3682L)
  dirs <- c("initializers", "checkpoints")
  for (d in dirs) dir.create(d, showWarnings = FALSE)
  engine_files <- list.files("package/R", "[.]R$", full.names = TRUE)
  hashes <- tools::md5sum(c(input, "networks.R", "engine-fit-batches.R", engine_files))
  ctl <- load_engine()$.gltbnn_control(list())
  run_id <- list(hashes = hashes, R = as.character(getRversion()), RNG = RNGkind(), control = ctl)
  if (file.exists("protocol.rds")) {
    stopifnot(identical(readRDS("protocol.rds"), run_id))
  } else atomic_rds(run_id, "protocol.rds")

  design <- expand.grid(width = c(25L, 10L), depth = c(10L, 5L, 2L),
                        fold = seq_len(5L), nuisance = c("ps", "m0", "m1"),
                        stringsAsFactors = FALSE)
  methods <- c("G", "GL", "GLT", "Unpenalized", "Dropout", "L1")
  warm <- transform(design, method = "Initial")
  warm$key <- with(warm, sprintf("f%d_%s_d%d_w%d", fold, nuisance, depth, width))
  warm$file <- file.path("initializers", paste0(warm$key, ".rds"))
  fits <- merge(design, data.frame(method = methods), by = NULL)
  fits <- fits[with(fits, order(-width, -depth, match(method, c("GLT","G","GL","Dropout","L1","Unpenalized")))),]
  fits$key <- with(fits, sprintf("f%d_%s_d%d_w%d_%s", fold, nuisance, depth, width, method))
  fits$file <- file.path("checkpoints", paste0(fits$key, ".rds"))
  pending <- function(grid) {
    found <- file.exists(grid$file)
    for (f in grid$file[found]) stopifnot(identical(readRDS(f)$run_id, run_id))
    grid[!found,, drop = FALSE]
  }
  phases <- list(pending(warm), pending(fits))
  if (any(vapply(phases, nrow, integer(1)) > 0)) {
    cl <- parallel::makePSOCKcluster(workers, outfile = "training.log")
    on.exit(parallel::stopCluster(cl))
    parallel::clusterExport(cl, c("data", "run_id"), envir = environment())
    parallel::clusterEvalQ(cl, {source("networks.R"); engine <- load_engine(); NULL})
    for (grid in phases) {
      if (!nrow(grid)) next
      ok <- parallel::parLapplyLB(cl, split(grid, seq_len(nrow(grid))), function(job) {
        tryCatch(fit_job(job, data, engine, run_id), error = function(e) {
          writeLines(conditionMessage(e), paste0(job$file, ".error"))
          FALSE
        })
      }, chunk.size = 1L)
      if (!all(unlist(ok))) stop("An individual fit failed. Inspect its .error file.")
    }
  }
  stopifnot(all(file.exists(fits$file)))
  configs <- unique(fits[c("width","depth","method","fold")])
  rows <- vector("list", nrow(configs))
  for (i in seq_len(nrow(configs))) {
    config <- configs[i,]
    test <- data$row_id[data$fold == config$fold]
    out <- data.frame(row_id = test, fold = config$fold, method = config$method,
                      depth = config$depth, width = config$width)
    for (role in c("m0", "m1", "ps")) {
      key <- with(config, sprintf("f%d_%s_d%d_w%d_%s", fold, role, depth, width, method))
      record <- readRDS(file.path("checkpoints", paste0(key, ".rds")))
      stopifnot(identical(record$row_id, test), identical(record$run_id, run_id))
      nm <- switch(role, m0 = "m0_hat", m1 = "m1_hat", ps = "e_hat")
      out[[nm]] <- record$prediction
    }
    rows[[i]] <- out
  }
  assembled <- do.call(rbind, rows)
  stopifnot(nrow(assembled) == 36L * nrow(data), !anyNA(assembled))
  if (file.exists("predictions.csv.gz")) {
    saved <- read.csv("predictions.csv.gz")
    stopifnot(isTRUE(all.equal(saved, assembled, check.attributes = FALSE)))
    cat("Verified all existing out-of-fold predictions.\n")
    return(invisible(TRUE))
  }
  gz <- gzfile("predictions.csv.gz.pending", open = "wt")
  write.csv(assembled, gz, row.names = FALSE)
  close(gz)
  stopifnot(file.rename("predictions.csv.gz.pending", "predictions.csv.gz"))
  cat("Completed 540 nuisance fits and saved all out-of-fold predictions.\n")
}
main()
