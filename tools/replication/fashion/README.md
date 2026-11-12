# T-shirts and pullovers in Fashion-MNIST

The study uses all 12,000 T-shirt/top and pullover images from the official training split and all 2,000 from the test split. Each 28 × 28 grayscale image gives 784 predictors. We center and scale each predictor using its training mean and sample standard deviation. The two classes have equal counts in each split. Original labels 0 and 2 become binary responses 0 (T-shirt/top) and 1 (pullover).

## Reproduce the 36 fits

Run these commands from this directory with R and the **digest** package. The optional website export also requires **jsonlite**.

```sh
Rscript --vanilla data/prepare.R
Rscript --vanilla compute/verify.R
Rscript --vanilla compute/run_full.R 4
Rscript --vanilla report/score.R data/data.rds results summary
Rscript --vanilla report/export_explorer.R data/data.rds results summary/scores.csv explorer
```

The integer following `run_full.R` sets the maximum number of parallel workers. Each worker has its own in-memory data. The same command resumes completed fits. The runner records source hashes and the R version.

We fit each method once at each architecture. Depth is two, five or ten hidden layers and width is ten or 25 units per layer. The six methods at a given architecture use the same weights after 600 unpenalized initialization updates and the same 600 subsequent minibatches, covering all 12,000 observations 50 times per phase. All networks have ReLU hidden units and one logit for the probability of pullover. The Bayesian networks average probabilities over 256 variational draws. The unpenalized and penalized networks predict with the final fitted weights, with dropout turned off.

The independent scorer checks the training schedule and held-out predictions, then reports accuracy and the mean negative natural log probability of the recorded class. The website export includes the first five test images of each class and their probabilities under every fit. The garment figure can be regenerated using `Rscript --vanilla figure/examples.R data/data.rds figure/examples.pdf`.

## Fit one model with the installed R package

```r
library(gltbnn)
garments <- readRDS("data/data.rds")
pixels <- function(x) matrix(as.integer(x) / 255, nrow = nrow(x))
y <- factor(garments$ytrain, levels = 0:1, labels = c("T-shirt/top", "pullover"))
fit <- gltbnn_fit(pixels(garments$Xtrain), y, outcome = "binary",
                  prior = "GLT", hidden = c(10L, 10L), seed = 241210L)
head(predict(fit, pixels(garments$Xtest), type = "prob", draws = 256L, seed = 251210L))
```

This example uses the package’s ordinary full-batch interface. The published comparison uses the minibatch procedure in `compute/run_full.R`.

Fashion-MNIST was introduced by Han Xiao, Kashif Rasul and Roland Vollgraf. The [original study](https://arxiv.org/abs/1708.07747), [dataset notice](data/LICENSE), [data description](data/README.md), and [package license](package/LICENSE.md) accompany the analysis.
