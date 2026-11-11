\name{gltbnn_fit}
\alias{gltbnn_fit}
\alias{print.gltbnn_fit}
\title{Fit a Bayesian neural network with grouped shrinkage priors}
\description{
Fit a fully connected neural network for a continuous response. Each hidden
layer uses a rectified linear unit (ReLU) activation. The final layer computes
an affine function and the response has Gaussian observation noise.
}
\usage{
gltbnn_fit(x, y, hidden = c(10L, 10L), prior = c("GLT", "GL", "G"),
           control = list(), seed = 1L, noise_prior = NULL)

\method{print}{gltbnn_fit}(x, ...)
}
\arguments{
  \item{x}{For fitting, a numeric matrix or numeric data frame with one row per
  observation and one column per predictor. Supply at least two observations
  and one predictor. For printing, a fitted \code{gltbnn_fit} object.}
  \item{y}{Numeric vector of responses, one per row of \code{x}. All training
  variables must be finite and have positive sample standard deviations.}
  \item{hidden}{Positive integer widths in input-to-output order. For example,
  \code{c(4L, 8L)} creates four neurons in the first hidden layer and eight in
  the second. Biases are included automatically.}
  \item{prior}{One of \code{"G"} (global), \code{"GL"} (grouped horseshoe),
  or \code{"GLT"} (global-local-tail, the default).}
  \item{control}{Named list of overrides for the fitting controls described below.}
  \item{seed}{Nonnegative integer random seed. The previous \code{.Random.seed}
  is restored after fitting.}
  \item{noise_prior}{Prior for the observation variance on the standardized
  response scale. The default \code{NULL} uses the paper's Jeffreys prior.
  For a proper inverse-gamma prior, supply a numeric vector named
  \code{shape} and \code{scale}, both finite and strictly positive. For
  example, \code{c(shape = 3, scale = 2)} has mean 1 in standardized units.}
  \item{...}{Additional arguments to the print method, currently unused.}
}
\details{
The incoming weights and bias of neuron \eqn{k} in hidden layer \eqn{l} form a
group of \eqn{d_l = K_{l-1} + 1}{d[l] = K[l-1] + 1} coefficients. Here
\eqn{K_l}{K[l]} is the width of layer \eqn{l} and \eqn{K_0}{K[0]} is the
number of predictors. Conditional on the scales, these coefficients have
independent zero-centered Gaussian priors with standard deviation
\eqn{\sigma\lambda_{kl}}{sigma * lambda[k,l]}. The quantity
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

With \code{"G"}, every neuron in a layer shares the scale
\eqn{\lambda_{kl}=\tau_l}{lambda[k,l] = tau[l]}. With \code{"GL"},
\eqn{\lambda_{kl}=\tau_l r_{kl}}{lambda[k,l] = tau[l] * r[k,l]}, where each
unit-specific multiplier has an independent standard half-Cauchy prior. Both
priors give \eqn{\tau_l}{tau[l]} a half-Cauchy prior with scale
\eqn{1/K_l}{1/K[l]}. With \code{"GLT"}, the unit scales follow a generalized
Pareto distribution conditional on the layer scale \eqn{\tau_l}{tau[l]} and
tail shape \eqn{\xi_l}{xi[l]}. The model learns both for each layer. Given
\eqn{\xi_l}{xi[l]}, the prior on \eqn{\tau_l}{tau[l]} is inverse-gamma with
shape \eqn{1+K_l/\xi_l}{1 + K[l]/xi[l]} and scale 1. The prior on
\eqn{\log\xi_l}{log(xi[l])} is standard normal truncated below at
\eqn{-\log 2}{-log(2)}.

All three models keep the chosen hidden widths and use a separate scale for
the final affine layer. Its weights and intercept have independent
zero-centered Gaussian priors conditional on a shared standard deviation
\eqn{\kappa}{kappa}, which has a half-Cauchy prior with scale 5.

The function centers each training variable and divides by its sample standard
deviation. It then initializes the network and fits a variational approximation
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
  Fitting starts at \code{log(0.01)} or at the nearest bound if that value
  falls outside the interval.}
  \item{\code{weight_draws}, \code{noise_draws}}{Monte Carlo draws per weight
  update (default 4) and per noise update (default 16).}
  \item{\code{monitor_draws}, \code{endpoint_draws}}{Monte Carlo draws for
  the evidence lower bound (ELBO) recorded during training (default 64) and at
  the independently sampled endpoint (default 256).}
}
With the Jeffreys prior, the objective uses the unnormalized joint density
with the arbitrary additive constant of the log prior set to zero. Under this
convention the endpoint value is a diagnostic for the variational fit.
Normalized model evidence requires a proper prior. The inverse-gamma option
includes the normalized log prior in the objective.
Each of the four training draw counts must be an even integer of at least two
because the estimator pairs each draw with its negative. With just one pair,
the Monte Carlo standard error is undefined. Training uses the requested
iteration budget.
}
\value{
An object of class \code{gltbnn_fit}. Fields include \code{prior}, \code{hidden},
the training dimensions \code{n} and \code{p}, \code{predictor_names},
the training centers and scales in \code{preprocess}, \code{control},
the chosen observation-variance prior in \code{noise_prior}, and
\code{seed}. The \code{engine} list contains the variational weight parameters
(\code{parameters}), scale factors (\code{state}), iteration history
(\code{trace}), and the final Monte Carlo ELBO estimate (\code{endpoint$value}).
Use \code{\link{gltbnn_shrinkage}} for unit and layer summaries in the
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
}
\seealso{\code{\link{gltbnn_shrinkage}}, \code{\link{predict.gltbnn_fit}}}
