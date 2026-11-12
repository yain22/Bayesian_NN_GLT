# From the study root: Rscript --vanilla compute/run_full.R 8
main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  workers <- if (length(args)) as.integer(args[1L]) else 1L
  stopifnot(length(args) <= 1L, is.finite(workers), workers >= 1L)
  Sys.setenv(OPENBLAS_NUM_THREADS = 1, OMP_NUM_THREADS = 1, MKL_NUM_THREADS = 1)
  source("compute/full_fit.R", local = .GlobalEnv)
  for (path in c("initializers", "results")) dir.create(path, showWarnings = FALSE)
  # The OS drops this loopback socket when the coordinator ends, even on interruption.
  lock <- serverSocket(27854L)
  on.exit(close(lock), add = TRUE)
  sources <- c("data/data.rds", "compute/run_full.R", "compute/full_fit.R", "compute/dropout.R",
               "compute/engine-fit-batches.R", list.files("package/R", full.names = TRUE))
  stopifnot(all(file.exists(sources)))
  protocol <- list(study = "Full native-resolution CIFAR-10 cat and dog", hashes = tools::md5sum(sources),
                   rng = RNGkind(), R = as.character(getRversion()))
  run_id <- cifar_hash(protocol)
  if (file.exists("protocol.rds")) {
    stopifnot(identical(protocol, readRDS("protocol.rds")))
  } else {
    cifar_save(protocol, "protocol.rds")
  }

  warm <- expand.grid(width = c(25L, 10L), depth = c(10L, 5L, 2L))
  warm$method <- "warmup"
  warm$path <- with(warm, cifar_init_path(depth, width))
  methods <- c("GLT", "GL", "G", "Unpenalized", "Dropout", "L1")
  fits <- merge(warm[c("depth", "width")], data.frame(method = methods), by = NULL)
  fits <- fits[order(match(fits$method, methods), -fits$width, -fits$depth), ]
  fits$path <- with(fits, sprintf("results/%s_d%d_w%d.rds", method, depth, width))
  pending <- function(grid) {
    done <- file.exists(grid$path)
    for (path in grid$path[done]) stopifnot(identical(readRDS(path)$run_id, run_id))
    grid[!done, , drop = FALSE]
  }
  phases <- list(pending(warm), pending(fits))
  if (!sum(vapply(phases, nrow, integer(1L)))) return(invisible(TRUE))
  cl <- parallel::makePSOCKcluster(min(workers, max(vapply(phases, nrow, integer(1L)))),
                                   outfile = "workers.log")
  on.exit(parallel::stopCluster(cl), add = TRUE)
  parallel::clusterExport(cl, "run_id", envir = environment())
  parallel::clusterEvalQ(cl, {
    source("compute/full_fit.R")
    source("compute/dropout.R")
    engine <- cifar_engine()
    data <- cifar_prepare(readRDS("data/data.rds"))
    gc()
    NULL
  })
  for (phase in phases) {
    if (!nrow(phase)) next
    jobs <- split(phase, seq_len(nrow(phase)))
    ok <- parallel::parLapplyLB(cl, jobs, function(job) {
      tryCatch({
        if (job$method == "warmup") {
          result <- cifar_warmup(data, job$depth, job$width, engine)
        } else {
          init <- readRDS(cifar_init_path(job$depth, job$width))
          stopifnot(identical(init$run_id, run_id))
          result <- cifar_fit(data, job$depth, job$width, job$method, init, engine)
        }
        result$run_id <- run_id
        cifar_save(result, job$path)
        if (file.exists(paste0(job$path, ".error"))) unlink(paste0(job$path, ".error"))
        TRUE
      }, error = function(e) {
        writeLines(conditionMessage(e), paste0(job$path, ".error"))
        FALSE
      })
    }, chunk.size = 1L)
    if (!all(unlist(ok))) stop("A job failed. Inspect the corresponding .error file.")
  }
  stopifnot(all(file.exists(fits$path)))
  writeLines("36 full-split CIFAR-10 cat and dog fits complete", "complete.txt")
}
main()
