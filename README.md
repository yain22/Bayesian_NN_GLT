<img src="docs/assets/mark.svg" align="right" width="140" alt="BNN GLT package mark">

# gltbnn

**Bayesian neural networks with shrinkage on individual hidden weights and biases.**

The R package accompanying *Variational Learning of Bayesian Neural Networks with a family of global–local shrinkage priors* fits fully connected ReLU networks under the global (G), horseshoe (GL), and global–local–tail (GLT) priors. Gaussian regression is the default. A Bernoulli likelihood handles binary outcomes and a categorical likelihood handles unordered classes.

[Package website](https://yain22.github.io/Bayesian_NN_GLT/) · [Explore a hidden unit](https://yain22.github.io/Bayesian_NN_GLT/illustration/) · [Explore predictive uncertainty](https://yain22.github.io/Bayesian_NN_GLT/prediction/)

## Installation

Download the source package from the [GitHub release](https://github.com/yain22/Bayesian_NN_GLT/releases/latest). In R 4.1 or later, run the following and choose the downloaded `.tar.gz` file.

```r
install.packages(file.choose(), repos = NULL, type = "source")
library(gltbnn)
```

## Fit a regression network

Use trunk diameter (the `Girth` column) and height to predict black cherry timber volume. Hold three trees out of training. Here `hidden = 4L` gives the network one hidden layer with four neurons.

```r
x <- trees[c("Girth", "Height")]
holdout <- c(4, 14, 24)

fit <- gltbnn_fit(
  x[-holdout, ], trees$Volume[-holdout],
  hidden = 4L, prior = "GLT", seed = 7L,
  noise_prior = c(shape = 2, scale = 1)
)
predict(fit, x[holdout, ])
predict(fit, x[holdout, ], type = "prediction_interval")
```

This example uses a proper inverse-gamma prior on the standardized observation variance. The first prediction estimates mean volume at each pair of measurements. The second gives a 95% variational predictive interval for an individual tree, including observation noise. Use `type = "mean_interval"` for uncertainty in the mean. Predictions are returned in the original response units.

Use `gltbnn_shrinkage(fit)` to inspect each incoming weight and bias. The returned columns identify the layer, receiving unit and incoming coordinate. [R reference manual](https://yain22.github.io/Bayesian_NN_GLT/reference.pdf)

## Fit a classifier

Set `outcome = "binary"` to model the log odds of the second response level or `outcome = "categorical"` to model multiple classes. In the categorical model the last level is the reference. Supply a factor to choose the level order. Each option returns posterior predictive class probabilities in that order.

```r
flowers <- droplevels(subset(iris, Species != "setosa"))
holdout <- seq(5, nrow(flowers), by = 5)
fit <- gltbnn_fit(flowers[-holdout, 1:4], flowers$Species[-holdout],
                  outcome = "binary", prior = "GL", hidden = 4L)
head(predict(fit, flowers[holdout, 1:4], type = "prob"))
```

- Inspect predicted probabilities of a malignant diagnosis on held-out Wisconsin Diagnostic Breast Cancer observations and vary the classification threshold. [Binary application and R code](https://yain22.github.io/Bayesian_NN_GLT/classification/#binary)
- Fit a categorical network to MNIST, select a held-out handwritten digit, and inspect its ten class probabilities. [MNIST application and R code](https://yain22.github.io/Bayesian_NN_GLT/classification/#mnist)
- Compare T-shirt and pullover probabilities on ten held-out Fashion-MNIST images and inspect accuracy and log loss across all 36 fits. [Fashion-MNIST application and replication code](https://yain22.github.io/Bayesian_NN_GLT/classification/fashion/)
- Trace a CIFAR-10 color pixel into a standardized network input, compare cat and dog probabilities, and inspect all 36 fits on the full cat/dog split. [CIFAR-10 application and replication code](https://yain22.github.io/Bayesian_NN_GLT/classification/cifar/)

## What each prior shrinks

For Gaussian regression, the package standardizes the training response and uses the Jeffreys prior $p(\sigma^2)\propto1/\sigma^2$ by default. Posterior propriety with this prior depends on the data. To use the supplement's proper inverse-gamma alternative, pass positive hyperparameters, for example `noise_prior = c(shape = 2, scale = 1)`.

For neuron $k$ in layer $l$, the vector $\mathbf{w}_{kl}$ contains its $d_l=K_{l-1}+1$ incoming weights, including its bias at $j=d_l$. Its scalar coefficient $w_{kl,j}$ has a zero-centered Gaussian prior with conditional standard deviation $\sigma\lambda_{kl,j}$ in regression and $\lambda_{kl,j}$ in classification. A small local scale concentrates the prior for that individual coefficient near zero. There are $M_l=K_l d_l$ such coefficients in layer $l$.

| `prior` | How the model sets the coefficient scale |
| --- | --- |
| `"G"` | Every coefficient in a layer shares $\lambda_{kl,j}=\tau_l$. |
| `"GL"` | Independent standard half-Cauchy multipliers $r_{kl,j}$ give $\lambda_{kl,j}=\tau_l r_{kl,j}$. |
| `"GLT"` | The coefficient scales follow a generalized Pareto distribution conditional on the layer scale $\tau_l$ and learned tail shape $\xi_l$. |

G and GL use a half-Cauchy layer scale with scale parameter $1/M_l$. For GLT, $\tau_l\mid\xi_l$ has an inverse-gamma prior with shape $1+M_l/\xi_l$ and scale 1.

Trace incoming connections in the architecture illustration, or compare mean-response and future-observation intervals from three fitted networks. [Architecture illustration](https://yain22.github.io/Bayesian_NN_GLT/illustration/) · [Prediction illustration](https://yain22.github.io/Bayesian_NN_GLT/prediction/)

## Simulation results

Select the signal design, training signal-to-noise ratio, depth and width to compare shrinkage among connections entering a neuron. The simulation explorer reports matched prediction errors and Monte Carlo standard errors across 100 replicates and provides the benchmark tables. [Explore the simulation results](https://yain22.github.io/Bayesian_NN_GLT/study/)

To reproduce one comparison, run the [C2 replication script](tools/replication/c2-additive.R) with the package installed. It generates the data, fits all six methods from a common initialization, and reports held-out errors and 90% interval coverage. With `jsonlite` installed, pass `--write-neuron-scales` from the repository root to regenerate the fitted-neuron panel’s individual weight and bias scales.

## License

The package is available under the [MIT license](LICENSE.md).
