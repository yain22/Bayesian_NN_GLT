<img src="docs/assets/mark.svg" align="right" width="140" alt="BNN GLT package mark">

# gltbnn

**Bayesian neural networks with grouped shrinkage priors for Gaussian regression.**

The R package accompanying *Variational Learning of Bayesian Neural Networks with a family of global–local shrinkage priors* fits fully connected ReLU networks under the global (G), grouped horseshoe (GL), and global–local–tail (GLT) priors. Choose a network width and a prior, fit the model, and predict a continuous response with uncertainty.

[Package website](https://yain22.github.io/Bayesian_NN_GLT/) · [Explore a hidden unit](https://yain22.github.io/Bayesian_NN_GLT/illustration/) · [Explore predictive uncertainty](https://yain22.github.io/Bayesian_NN_GLT/prediction/)

## Installation

Install from GitHub in R 4.1 or later. Run `install.packages("remotes")` once if you need the installer.

```r
remotes::install_github("yain22/Bayesian_NN_GLT")
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

Use `gltbnn_shrinkage(fit)` to inspect the fitted incoming-weight prior scale for every hidden unit. The [R reference manual](https://yain22.github.io/Bayesian_NN_GLT/reference.pdf) documents the variational scale summaries and predictive draws.

## What each prior shrinks

The study fits on the standardized response scale and uses the Jeffreys prior for the observation variance, $p(\sigma^2)\propto1/\sigma^2$. Posterior propriety with this default depends on the data. To use the supplement's proper inverse-gamma alternative, pass positive hyperparameters, for example `noise_prior = c(shape = 2, scale = 1)`. The hidden-unit prior still uses the same $\sigma$.

For neuron $k$ in layer $l$, the vector $\mathbf w_{kl}$ contains its incoming weights and bias, a total of $d_l$ coefficients. Each coefficient has a zero-centered Gaussian prior with conditional standard deviation $\sigma\lambda_{kl}$. The Gaussian observation model supplies the same noise standard deviation $\sigma$. A small $\lambda_{kl}$ pulls the whole group toward zero. Each of the $K_l$ neurons in the layer remains in the fitted architecture.

| `prior` | How the model sets the group scale |
| --- | --- |
| `"G"` | All neurons in a layer share one scale, $\lambda_{kl}=\tau_l$. |
| `"GL"` | Each neuron has a half-Cauchy multiplier $r_{kl}$, giving $\lambda_{kl}=\tau_l r_{kl}$. |
| `"GLT"` | Neuron scales follow a generalized Pareto distribution with layer scale $\tau_l$ and a learned layer-specific tail shape $\xi_l$. |

The architecture illustration lets you select a neuron, trace its incoming connections and explore how the scale changes the prior. The prediction illustration compares the three fitted priors on the same synthetic data and shows how uncertainty differs for a mean response and a new observation. [Architecture illustration](https://yain22.github.io/Bayesian_NN_GLT/illustration/) · [Prediction illustration](https://yain22.github.io/Bayesian_NN_GLT/prediction/)

## License

The package is available under the [MIT license](LICENSE.md).
