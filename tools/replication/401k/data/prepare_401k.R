# Prepare the public 1991 SIPP sample and the five common cross-fitting folds.
# R 3.6 or later and its recommended package foreign are sufficient.
# Usage: Rscript prepare_401k.R [output-directory]
args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) == 0L) "." else args[[1L]]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

commit <- "a8ef8ef6e48d7d7056669e08f0b812bdb6528106"
url <- paste0("https://raw.githubusercontent.com/VC2015/DMLonGitHub/", commit,
              "/sipp1991.dta")
source_file <- file.path(out_dir, "sipp1991.dta")
if (!file.exists(source_file)) {
    download.file(url, source_file, mode = "wb", quiet = TRUE)
}
stopifnot(unname(tools::md5sum(source_file)) == "537eba85ba6ec692740f82138c3f91b7")
sipp <- foreign::read.dta(source_file, convert.factors = FALSE)
covariates <- c("age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown")
input <- sipp[c("e401", "net_tfa", covariates)]
stopifnot(nrow(input) == 9915L, !anyNA(input), all(input$e401 %in% 0:1),
          sum(input$e401) == 3682L)

# Randomly permute the household indices within each eligibility stratum,
# then allocate them cyclically to folds, continuing the cycle between arms.
RNGkind(kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
set.seed(401015L)
fold <- integer(nrow(input))
offset <- 0L
for (eligible in 0:1) {
    ids <- which(input$e401 == eligible)
    ids <- sample(ids, length(ids), replace = FALSE)
    fold[ids] <- 1L + ((seq_along(ids) - 1L + offset) %% 5L)
    offset <- (offset + length(ids)) %% 5L
}
stopifnot(all(table(fold) == 1983L))
analysis <- data.frame(row_id = seq_len(nrow(input)), fold = fold, input)
write_analysis <- function(data, filename) {
    con <- file(filename, open = "wb")
    on.exit(close(con))
    write.table(data, con, quote = FALSE, sep = ",", row.names = FALSE,
                col.names = TRUE, eol = "\r\n")
}
write_analysis(analysis, file.path(out_dir, "analysis.csv"))
