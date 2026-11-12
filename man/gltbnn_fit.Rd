\name{gltbnn_fit}
\alias{gltbnn_fit}
\alias{print.gltbnn_fit}
\title{Fit a Bayesian neural network with shrinkage priors}
\description{
Fit a fully connected neural network for Gaussian regression, binary logistic
classification or categorical softmax classification. Each hidden layer uses
a rectified linear unit (ReLU) activation.
}
\usage{
gltbnn_fit(x, y, hidden = c(10L, 10L), prior = c("GLT", "GL", "G"),
           control = list(), seed = 1L, noise_prior = NULL,
           outcome = c("continuous", "binary", "categorical"),
           geometry = c("coefficient", "unit"))

\method{print}{gltbnn_fit}(x, ...)
}
\arguments{
  \item{x}{For fitting, a numeric matrix or numeric data frame with one row per
  observation and one column per predictor. Supply at least two observations
  and one predictor. For printing, a fitted \code{gltbnn_fit} object.}
  \item{y}{One response per row of \code{x}. For regression, supply finite
  numeric responses with a positive sample standard deviation. For classification,
  supply a factor or finite numeric, logical or nonempty character labels.
  Every declared factor level must occur in training. Factor levels set the
  class order. Other label vectors use the level order of \code{factor(y)}.}
  \item{hidden}{Positive integer widths in input-to-output order. For example,
  \code{c(4L, 8L)} creates four neurons in the first hidden layer and eight in
  the second. Biases are included automatically.}
  \item{prior}{One of \code{"G"} (global), \code{"GL"} (horseshoe),
  or \code{"GLT"} (global-local-tail, the default).}
  \item{control}{Named list of overrides for the fitting controls described below.}
  \item{seed}{Nonnegative integer random seed. The previous \code{.Random.seed}
  is restored after fitting.}
  \item{noise_prior}{Prior for the observation variance on the standardized
  response scale in continuous regression. The default \code{NULL} uses
  the paper's Jeffreys prior. Use \code{NULL} for classification.
  For a proper inverse-gamma prior, supply a numeric vector named
  \code{shape} and \code{scale}, both finite and strictly positive. For
  example, \code{c(shape = 3, scale = 2)} has mean 1 in standardized units.}
  \item{outcome}{\code{"continuous"} (the default), \code{"binary"} for
  exactly two classes, or \code{"categorical"} for two or more nominal classes.}
  \item{geometry}{\code{"coefficient"} (the default) gives each hidden weight
  and bias its own local scale under GL and GLT. Use \code{"unit"} to fit the
  earlier grouped specification, which shares one local scale across each
  unit's incoming coefficients. G has one common scale per layer in both
  specifications, with different prior calibration as described below.}
  \item{...}{Additional arguments to the print method, currently unused.}
}
\details{
The incoming weights and bias of neuron \eqn{k} in hidden layer \eqn{l} form a
vector \eqn{\mathbf{w}_{kl}}{w[k,l]} of
\eqn{d_l = K_{l-1} + 1}{d[l] = K[l-1] + 1} coefficients. Here
\eqn{K_l}{K[l]} is the width of layer \eqn{l}, \eqn{K_0}{K[0]} is the
number of predictors used by the network, and \eqn{M_l=K_ld_l}{M[l] = K[l] * d[l]}
counts its scalar hidden coefficients, including biases. In Gaussian regression,
coefficient \eqn{w_{kl,j}}{w[k,l,j]} has a zero-centered Gaussian prior with
conditional standard deviation \eqn{\sigma\lambda_{kl,j}}{sigma * lambda[k,l,j]}.
The final coordinate \eqn{j=d_l}{j = d[l]} is the unit's bias. The weights
are independent given their scales and the observation variance. The quantity
\eqn{\sigma^2}{sigma^2} is the Gaussian observation noise variance on the
standardized training scale. The same \eqn{\sigma}{sigma} sets the units
for the hidden weight prior. The default uses the improper density kernel
\eqn{1/\sigma^2}{1/sigma^2}. Its posterior is proper almost surely when the
responses have a joint density conditional on the predictors. Some
nonconstant data can yield an improper posterior. For example, under G with
one hidden layer, three or more distinct univariate inputs and exactly affine
responses with nonzero slope can have this property. Interpret variational
draws as approximate posterior draws when the posterior is proper.

The alternative inverse-gamma prior gives a proper posterior for every finite
dataset accepted by this function. For variance \eqn{s}, shape \eqn{a} and
scale \eqn{b}, its density is proportional to
\eqn{s^{-(a+1)}\exp(-b/s)}{s^(-a-1) * exp(-b/s)}. The prior uses
standardized fitting units. Its scale parameter on the original response
scale equals \eqn{b} times the training sample variance of \code{y}.

With \code{"G"}, every coefficient in a layer shares the scale
\eqn{\lambda_{kl,j}=\tau_l}{lambda[k,l,j] = tau[l]}. With \code{"GL"},
\eqn{\lambda_{kl,j}=\tau_l r_{kl,j}}{lambda[k,l,j] = tau[l] * r[k,l,j]},
where the multipliers are independent standard half-Cauchy variables. Both
priors give \eqn{\tau_l}{tau[l]} a half-Cauchy prior with scale
\eqn{1/M_l}{1/M[l]}. With \code{"GLT"}, the coefficient scales independently
follow a generalized Pareto distribution conditional on the layer scale \eqn{\tau_l}{tau[l]} and
tail shape \eqn{\xi_l}{xi[l]}. The model learns both for each layer. Given
\eqn{\xi_l}{xi[l]}, the prior on \eqn{\tau_l}{tau[l]} is inverse-gamma with
shape \eqn{1+M_l/\xi_l}{1 + M[l]/xi[l]} and scale 1. The prior on
\eqn{\log\xi_l}{log(xi[l])} is standard normal truncated below at
\eqn{-\log 2}{-log(2)}.

With \code{geometry = "unit"}, all \eqn{d_l}{d[l]} incoming coordinates of
unit \eqn{k} share the scalar \eqn{\lambda_{kl}}{lambda[k,l]}. There are
\eqn{K_l}{K[l]} units per layer, so the earlier specification uses
\eqn{1/K_l}{1/K[l]} as the G/GL half-Cauchy scale and
\eqn{1+K_l/\xi_l}{1 + K[l]/xi[l]} as the GLT inverse-gamma shape.

All three models keep the chosen hidden widths and use a separate scale for
the final affine layer. Its weights and intercept have independent
zero-centered Gaussian priors conditional on a shared standard deviation
\eqn{\kappa}{kappa}, which has a half-Cauchy prior with scale 5.

For classification, the conditional prior on each incoming weight or bias
has standard deviation \eqn{\lambda_{kl,j}}{lambda[k,l,j]} in the default geometry. The classification
likelihood has no observation-variance parameter. Binary classification uses
one affine logit for the probability of the second level. For \eqn{C}
categorical levels the last is the reference, with logit zero. Each of the
first \eqn{C-1} levels has an affine logit for its log odds relative to the
reference. Softmax converts these logits to class probabilities. All free
readout coefficients share the scale \eqn{\kappa}{kappa}; the prior is
specified on contrasts with the chosen reference class.

The function centers predictors and divides by their training sample standard
deviations. Regression also standardizes the response and requires every
predictor to vary. Classification omits predictors constant in training and
requires at least one varying predictor. Supply the original set of columns
at prediction time. The function fits a variational approximation
with independent Gaussian weight factors. The entries below are available in
\code{control}.
\describe{
  \item{\code{init_steps}}{Number of deterministic-network optimizer steps
  used to initialize the weight means (default 600).}
  \item{\code{outer}, \code{weight_steps}}{Number of outer iterations
  (default 60) and weight updates per iteration (default 10).}
  \item{\code{lr}, \code{grad_clip}}{Initial learning rate for each
  variational weight update (default 0.003) and gradient norm cap (default 10).}
  \item{\code{max_backtracks}}{Maximum number of learning-rate halvings per
  weight update (default 10).}
  \item{\code{logsd_bounds}}{Lower and upper bounds on the log standard
  deviations of the variational weight factors (default \code{c(-15, 3)}).
  At both endpoints, \code{exp(2 * bound)} must give a positive finite variance
  in double precision.
  Fitting starts at \code{log(0.01)} or at the nearest bound if that value
  falls outside the interval.}
  \item{\code{weight_draws}, \code{noise_draws}}{Monte Carlo draws per weight
  update (default 4) and per Gaussian observation-variance update (default 16).
  Classification uses only the weight draws in this pair.}
  \item{\code{monitor_draws}, \code{endpoint_draws}}{Monte Carlo draws for
  the evidence lower bound (ELBO) recorded during training (default 64) and at
  the independently sampled endpoint (default 256).}
}
With the Jeffreys prior, the objective uses the unnormalized joint density
with the arbitrary additive constant of the log prior set to zero. Under this
convention the endpoint value is a diagnostic for the variational fit.
Normalized model evidence requires a proper prior. The inverse-gamma option
includes the normalized log prior in the objective. Its component
\code{noise_neg_kl} combines the expected log prior and factor entropy as
minus the KL divergence from the fitted variance factor to its prior.
The Jeffreys fit records \code{noise_prior} and \code{noise_entropy} separately.
Each of the four training draw counts must be an even integer of at least two
because the estimator pairs each draw with its negative. With just one pair,
the Monte Carlo standard error is undefined. Training uses the requested
iteration budget.
}
\value{
An object of class \code{gltbnn_fit}. Fields include \code{outcome}, \code{geometry},
\code{prior}, \code{hidden},
the training dimensions \code{n} and \code{p}, \code{predictor_names},
the training centers and scales in \code{preprocess}, \code{control},
the chosen observation-variance prior in \code{noise_prior}, and
\code{seed}. Classifiers store the class order in \code{levels} and the indices
of retained predictors in \code{preprocess$x_keep}.
The \code{engine} list contains the variational weight parameters
(\code{parameters}), scale factors (\code{state}), iteration history
(\code{trace}), and the final Monte Carlo ELBO estimate (\code{endpoint$value}).
Use \code{\link{gltbnn_shrinkage}} for coefficient and layer summaries in the
paper's notation. In the stored factors, \code{engine$state$g[[l]]$tau}
represents \eqn{\tau_l^2}{tau[l]^2} under G and GL, and
\eqn{\tau_l}{tau[l]} under GLT. For GLT, \code{g[[l]]$tail$r} is
the variational expectation of \eqn{1/\xi_l}{1/xi[l]} and
\code{g[[l]]$tail$eta_mean} is the expectation of \eqn{\log\xi_l}{log(xi[l])}.
Inverse-gamma factors store their shape in \code{a} and their scale in
\code{b}. The reciprocal has a gamma distribution with shape \code{a} and
rate \code{b}.
The print method returns its input invisibly.
}
\examples{
x <- data.frame(z = seq(-1, 1, length.out = 21))
y <- x$z^2 + 0.1 * cos(9 * x$z)
# A short run to illustrate the interface.
fit <- gltbnn_fit(x, y, hidden = 3L, prior = "GL",
                  control = list(init_steps = 10L, outer = 2L, weight_steps = 2L))
fit
new_x <- data.frame(z = c(-0.5, 0, 0.5))
predict(fit, new_x)
predict(fit, new_x, type = "prediction_interval", level = 0.9)

# Probability of each species, with species labels retained in the columns.
flowers <- datasets::iris
held_out <- seq(5, nrow(flowers), by = 5)
species_fit <- gltbnn_fit(flowers[-held_out, 1:4], flowers$Species[-held_out],
                          outcome = "categorical", hidden = 4L, prior = "GL",
                          control = list(init_steps = 10L, outer = 2L,
                                         weight_steps = 2L))
head(predict(species_fit, flowers[held_out, 1:4], type = "prob", draws = 32L))
}
\seealso{\code{\link{gltbnn_shrinkage}}, \code{\link{predict.gltbnn_fit}}}
