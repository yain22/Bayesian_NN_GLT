\name{gltbnn_shrinkage}
\alias{gltbnn_shrinkage}
\title{Inspect shrinkage for each hidden unit}
\description{
Summarize the fitted scale shared by the incoming weights and bias of each
hidden unit. A smaller typical conditional prior standard deviation means
these coefficients are more tightly concentrated around zero under their
conditional prior.
}
\usage{gltbnn_shrinkage(fit)}
\arguments{
  \item{fit}{A model returned by \code{\link{gltbnn_fit}}.}
}
\value{
A data frame with one row per hidden unit, ordered by layer and unit.
\describe{
  \item{\code{layer}, \code{unit}}{One-based indices of the hidden layer
  \eqn{l} and the receiving unit \eqn{k} within that layer.}
  \item{\code{group_size}}{Number of coefficients entering the unit,
  including its bias.}
  \item{\code{scale_precision}}{Expected inverse squared unit scale
  \eqn{E_q[\lambda_{kl}^{-2}]}{E_q[1/lambda_kl^2]} under the fitted
  variational distribution \eqn{q}. This excludes the observation-noise factor.}
  \item{\code{geometric_prior_sd}}{Geometric mean under \eqn{q} of the
  conditional prior standard deviation of an incoming coefficient,
  \eqn{\exp\{E_q[\log(\sigma\lambda_{kl})]\}}{exp(E_q[log(sigma * lambda_kl)])},
  on the standardized fitting scale.}
  \item{\code{geometric_layer_scale}}{Geometric mean of the layer scale,
  \eqn{\exp\{E_q[\log\tau_l]\}}{exp(E_q[log(tau_l)])}.}
  \item{\code{geometric_tail_shape}}{For GLT, the geometric mean of the
  layer's generalized Pareto tail shape,
  \eqn{\exp\{E_q[\log\xi_l]\}}{exp(E_q[log(xi_l)])}. G and GL have no
  fitted tail shape and return \code{NA_real_}.}
}
Both layer summaries repeat on every row from the same layer.
}
\details{
Conditional on the scale and on the observation-noise variance
\eqn{\sigma^2}{sigma^2}, each incoming coefficient has a zero-mean Gaussian
prior with variance \eqn{\sigma^2\lambda_{kl}^2}{sigma^2 * lambda_kl^2}.
The fit assigns the observation-noise variance its own variational factor,
independent of the scale factors under \eqn{q}. The expected conditional
prior precision is therefore
\eqn{E_q[\sigma^{-2}] E_q[\lambda_{kl}^{-2}]}{E_q[1/sigma^2] * E_q[1/lambda_kl^2]}.
The geometric prior standard deviation uses the expected log standard
deviation. The variational posterior standard deviation
of the weight itself comes from its separate Gaussian weight factor.

Under G all units in a layer share one scale and have identical summaries.
Under GL the unit scale is the product of the layer and local scales. Under GLT
the unit scales follow a generalized Pareto distribution given the layer scale
and tail shape.
The inverse-gamma scale factor is fitted to \eqn{\tau_l^2}{tau_l^2} under
G and GL and to \eqn{\tau_l}{tau_l} under GLT. The reported layer scale
uses \eqn{\tau_l}{tau_l} itself in all three models.
}
\examples{
x <- data.frame(z = seq(-1, 1, length.out = 15))
y <- x$z^2
fit <- gltbnn_fit(x, y, hidden = 3L, prior = "GLT",
                  control = list(init_steps = 10L, outer = 2L, weight_steps = 2L))
units <- gltbnn_shrinkage(fit)
units
unique(units[c("layer", "geometric_layer_scale", "geometric_tail_shape")])
}
\seealso{\code{\link{gltbnn_fit}}}
