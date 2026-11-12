# Draw the first five test photographs of each class in the original file order.
library(grid)
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
data <- readRDS(args[1L])
stopifnot(identical(dim(data$Xtest), c(2000L, 784L)), is.raw(data$Xtest),
          all(diff(data$test_index) > 0), identical(sort(unique(data$ytest)), 0:1))
pdf(args[2L], width = 5.1, height = 1.45, family = "Times", useDingbats = FALSE,
    colormodel = "gray", title = "Fashion-MNIST test examples")
grid.newpage()
for (k in 0:1) {
  ids <- head(which(data$ytest == k), 5L)
  y <- unit(c(1.075, .375)[k + 1L], "inches")
  grid.text(c("T-shirt/top", "Pullover")[k + 1L], x = unit(.04, "inches"), y = y,
            just = "left", gp = gpar(fontsize = 11))
  for (j in seq_along(ids)) {
    pixel <- matrix(as.integer(data$Xtest[ids[j], ]), nrow = 28L, byrow = TRUE)
    grid.raster(as.raster(pixel / 255), x = unit(1.23 + .85 * (j - 1), "inches"), y = y,
                width = unit(.64, "inches"), height = unit(.64, "inches"), interpolate = FALSE)
  }
}
invisible(dev.off())
