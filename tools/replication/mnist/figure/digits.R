# Draw the first two occurrences of each digit in the official MNIST test file.
library(grid)
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
directory <- dirname(normalizePath(script))
data <- readRDS(file.path(directory, "..", "data", "data.rds"))
stopifnot(identical(data$test_index, seq_len(10000L)), is.raw(data$Xtest),
          identical(dim(data$Xtest), c(10000L, 784L)))
chosen <- vapply(0:9, function(k) head(which(data$ytest == k), 2L), integer(2L))

draw_digits <- function() {
  pdf(file.path(directory, "mnist_digits.pdf"), width = 6.44, height = 1.44,
      title = "Handwritten digits from the official MNIST test set", family = "Times",
      colormodel = "gray", useDingbats = FALSE)
  on.exit(dev.off())
  grid.newpage()
  position <- function(x) unit(x, "inches")
  label <- function(text, x, y, just = "centre") {
    grid.text(text, position(x), position(y), just = just,
              gp = gpar(fontsize = 10, fontfamily = "Times"))
  }
  label("Digit", .025, 1.33, "left")
  label("First", .025, .89, "left")
  label("Second", .025, .29, "left")
  for (k in 0:9) {
    xpos <- .99 + .575 * k
    label(k, xpos, 1.33)
    for (row in 1:2) {
      i <- chosen[row, k + 1L]
      pixels <- matrix(as.integer(data$Xtest[i, ]), 28L, byrow = TRUE) / 255
      grid.raster(as.raster(pixels), x = position(xpos), y = position(c(.89, .29)[row]),
                  width = position(.506), height = position(.506), interpolate = FALSE)
    }
  }
}
draw_digits()
