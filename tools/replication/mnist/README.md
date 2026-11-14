# MNIST handwritten digits · Table 9

All six methods train on the official 60,000 handwritten digits and predict the labels of the 10,000 official test images. The [36 published scores](summary/scores.csv) give the number classified correctly, accuracy and mean log loss for each combination of method, depth and width. See the [test gallery](../../../docs/classification/mnist-full/test-digits.png) and [data notes](data/README.md) for the image source and pixel order.

## Inspect the saved fits

Save the [saved results archive](../../../docs/classification/mnist-full/results.zip) as `results.zip` in this study directory (`tools/replication/mnist/`). In R, set the working directory to that folder and run `unzip("results.zip")`. The archive contains the original 36 fits, six shared initializations and the fitting protocol. For example, inspect the ten probabilities for the first official test image in the most accurate fit:

```r
fit <- readRDS("results/GLT_d2_w25.rds")
i <- 1L
data.frame(test_record = fit$test_id[i], digit = fit$truth[i],
           prediction = fit$pred[i])
setNames(round(fit$prob[i, ], 3), 0:9)
```

## Rescore the 36 saved fits

Use R 4.1 or later with the **digest** package installed. After unpacking the saved fits, run these commands in a shell from `tools/replication/mnist/`. They prepare the images, verify the implementation, and score the saved predictions. The preparation script downloads the four MNIST files from the Computer Vision Data Foundation mirror and checks their digests before converting them to R matrices.

```sh
Rscript --vanilla data/prepare.R
Rscript --vanilla compute/verify.R
Rscript --vanilla report/score.R data/data.rds results checked
```

The scorer writes `checked/scores.csv` and the editable manuscript table. It checks the test records, shared initializers and training schedules, reconstructs each frequentist prediction, and verifies the Bayesian probabilities on the first test image of each digit. To regenerate the manuscript gallery, run `Rscript --vanilla figure/digits.R`.

## Train all 36 networks from scratch

Use a fresh copy of the study directory with no extracted fit archive. Run these four commands from that directory with R 4.1 or later and **digest** installed:

```sh
Rscript --vanilla data/prepare.R
Rscript --vanilla compute/verify.R
Rscript --vanilla compute/run_full.R 1
Rscript --vanilla report/score.R data/data.rds results checked
```

The integer sets the maximum number of workers. The full study uses the bundled gltbnn 0.2.0 source and the study's categorical minibatch extension. The 717 training-varying pixels are standardized using training statistics. At each architecture the methods start from the same weights after 600 unpenalized updates and receive 600 further updates or proposals. Each BNN averages predictive probabilities over 256 weight draws. A fit's predicted class is the digit with the greatest probability, and its log loss is the mean negative log probability of the observed digit.

[Package license](package/LICENSE.md) · [MNIST image attribution and license](data/README.md)
