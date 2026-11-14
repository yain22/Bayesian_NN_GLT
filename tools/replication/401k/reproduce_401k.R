# Run from the extracted 401(k) replication directory.
data_401k <- function() {
  input <- file.path("data", "analysis.csv")
  # The two digests allow the original CRLF and conventional LF line endings.
  stopifnot(file.exists(input), unname(tools::md5sum(input)) %in% c(
    "cf7111bd33cfe9fd1a58784ad44400c1", "b3bb5a4eea35387281a193f34fd19205"))
  read.csv(input)
}

fit_401k_glt_fold <- function(d, k, hidden = rep(25L, 10L)) {
  stopifnot(length(k) == 1L, k %in% 1:5, length(hidden) %in% c(2L, 5L, 10L),
            all(hidden %in% c(10L, 25L)), length(unique(hidden)) == 1L,
            nrow(d) == 9915L, identical(d$row_id, seq_len(nrow(d))),
            all(table(factor(d$fold, levels = 1:5)) == 1983L))
  # Load exactly the implementation and minibatch schedule used by the full grid.
  original <- setwd("fit")
  on.exit(setwd(original))
  source("networks.R", local = TRUE)
  engine <- load_engine()
  scratch <- tempfile("glt-401k-fold-")
  dir.create(scratch)
  setwd(scratch)
  on.exit({
    setwd(original)
    unlink(scratch, recursive = TRUE)
  })
  dir.create("initializers")
  dir.create("checkpoints")
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")

  rows <- d$row_id[d$fold == k]
  pred <- data.frame(row_id = rows)
  for (role in c("m0", "m1", "ps")) {
    job <- list(fold = k, nuisance = role, depth = length(hidden),
                width = as.integer(hidden[1L]), method = "Initial")
    fit_job(job, d, engine, run_id = "example")
    job$method <- "GLT"
    fit_job(job, d, engine, run_id = "example")
    name <- sprintf("f%d_%s_d%d_w%d_GLT.rds", k, role, job$depth, job$width)
    fit <- readRDS(file.path("checkpoints", name))
    stopifnot(identical(fit$row_id, rows))
    column <- if (role == "ps") "e" else role
    pred[[column]] <- fit$prediction
  }
  pred
}
