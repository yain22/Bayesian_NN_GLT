# CIFAR-10 cat and dog classification

This example compares G, GL, GLT, unpenalized, dropout, and L₁ networks with two, five, or ten hidden layers and ten or 25 units per layer. Each of the 36 fits uses 5,000 cats and 5,000 dogs from the official training split. Each is evaluated on the same 1,000 cats and 1,000 dogs from the official test split.

From this directory, use R with the **digest** package installed. The optional explorer export also uses **jsonlite**.

```sh
Rscript --vanilla data/prepare.R
Rscript --vanilla compute/verify.R
Rscript --vanilla compute/run_full.R 4
Rscript --vanilla report/score.R data/data.rds results summary
Rscript --vanilla report/export_explorer.R data/data.rds results summary/scores.csv explorer
```

The integer after `run_full.R` selects the maximum number of parallel workers. Each worker holds its own standardized data matrices in memory. Restart with the same command to reuse completed fits. The runner records the R version, RNG configuration and source hashes when it starts. To refit with different code or settings, extract the code into a fresh directory.

`data/prepare.R` downloads the CIFAR-10 binary archive, checks its MD5 digest, and selects cats and dogs in their original order. The 3,072 coordinates are the red, green, and blue channel blocks, each read by row. All fits use the mean and standard deviation of each coordinate across the official training images for standardization. See the [data description](data/README.md).

Each network uses ReLU hidden units and a single logit for the probability of dog. G assigns one Gaussian scale to each hidden layer. GL and GLT assign a local scale to each scalar hidden weight and bias. The shared initialization uses 600 unpenalized Adam updates. Each method then takes 600 scheduled minibatches in the same order at that architecture. Each phase comprises 60 complete passes, in batches of 1,024 with a final batch of 784. FNN dropout uses probability 0.2 and the L₁ coefficient is 0.001. The BNN predictions average the class probabilities over 256 independent draws from the fitted variational distribution.

`package/R` is the frozen fitting source used for this experiment. `compute/engine-fit-batches.R` implements the minibatch weight update. `compute/verify.R` checks its objective and gradient against the full-batch implementation. `report/score.R` independently checks the schedules, verifies the data and replays predictions before writing the 36 accuracies and mean log losses to `summary/scores.csv`. The optional export writes the selected photographs, probabilities, and training pixel statistics for the interactive illustration.

### Fitting one model through the R package

After preparing the split, the installed package also accepts these images through its ordinary full-batch binary interface. Supply a factor with the cat level first and dog second. The function standardizes each column using its training observations.

```r
library(gltbnn)
photos <- readRDS("data/data.rds")
as_pixels <- function(bytes) matrix(as.integer(bytes) / 255, nrow = nrow(bytes))
label <- factor(photos$ytrain, levels = 0:1, labels = c("cat", "dog"))
fit <- gltbnn_fit(as_pixels(photos$Xtrain), label, outcome = "binary",
                  prior = "GLT", hidden = c(10L, 10L), seed = 241210L)
prob <- predict(fit, as_pixels(photos$Xtest), type = "prob", draws = 256L, seed = 251210L)
head(prob)
```

For the reported 36-fit benchmark, use the minibatch runner above.

The code is distributed under the [MIT license](package/LICENSE.md). The photographs are from CIFAR-10, compiled by Alex Krizhevsky, Vinod Nair and Geoffrey Hinton. [Dataset and binary format](https://www.cs.toronto.edu/~kriz/cifar.html).
