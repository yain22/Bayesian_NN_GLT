\name{gltbnn_shrinkage}
\alias{gltbnn_shrinkage}
\title{Inspect the shrinkage of hidden weights and biases}
\description{
Summarize the conditional prior standard deviation for each hidden coefficient.
The variational distribution gives a fitted distribution for its scale;
the geometric mean summarizes the typical prior standard deviation.
}
\usage{gltbnn_shrinkage(fit)}
\arguments{
  \item{fit}{A model returned by \code{\link{gltbnn_fit}}.}
}
\value{
A data frame ordered by hidden layer, receiving unit and incoming coordinate.
\describe{
  \item{\code{layer}, \code{unit}}{One-based indices of the hidden layer
  \eqn{l} and receiving unit \eqn{k}.}
  \item{\code{coordinate}, \code{is_bias}}{For the default coefficient geometry,
  the one-based incoming coordinate \eqn{j} and a logical indicator of the
  bias. The last coordinate \eqn{d_l=K_{l-1}+1}{d[l] = K[l-1] + 1} multiplies
  the constant input 1.}
  \item{\code{group_size}}{Number of coefficients represented by the row.
  This is one for coefficient geometry. For a fit with \code{geometry = "unit"}
  or a saved fit from the original package release, the result has one row per
  hidden unit, with \eqn{d_l}{d[l]} coordinates in each group and the original
  set of columns.}
  \item{\code{scale_precision}}{Expected inverse squared scale
  \eqn{E_q[\lambda_{kl,j}^{-2}]}{E_q[1/lambda[k,l,j]^2]} under the fitted
  variational distribution \eqn{q}. In Gaussian regression, multiply by
  \eqn{E_q[\sigma^{-2}]}{E_q[1/sigma^2]} for the expected conditional prior precision.}
  \item{\code{geometric_prior_sd}}{For continuous outcomes, the geometric mean
  \eqn{\exp\{E_q[\log(\sigma\lambda_{kl,j})]\}}{exp(E_q[log(sigma * lambda[k,l,j])])}
  in standardized fitting units. For binary or categorical outcomes this is
  \eqn{\exp\{E_q[\log\lambda_{kl,j}]\}}{exp(E_q[log(lambda[k,l,j])])}.}
  \item{\code{geometric_layer_scale}}{Geometric mean of the layer scale,
  \eqn{\exp\{E_q[\log\tau_l]\}}{exp(E_q[log(tau[l])])}.}
  \item{\code{geometric_tail_shape}}{For GLT, the geometric mean of the layer
  tail shape, \eqn{\exp\{E_q[\log\xi_l]\}}{exp(E_q[log(xi[l])])}.
  G and GL return \code{NA_real_}.}
}
Both layer summaries repeat on every row from the same layer.
}
\details{
In the default geometry, each component \eqn{w_{kl,j}}{w[k,l,j]} of the
incoming vector \eqn{\mathbf{w}_{kl}}{w[k,l]} has conditional prior variance
\eqn{\sigma^2\lambda_{kl,j}^2}{sigma^2 * lambda[k,l,j]^2} in regression and
\eqn{\lambda_{kl,j}^2}{lambda[k,l,j]^2} in classification. A small fitted
scale concentrates that coefficient's conditional prior near zero. Under G
all coefficients in a layer share \eqn{\tau_l}{tau[l]}.
Under GL the scale is the product of the layer scale and the coefficient's
local multiplier. Under GLT the coefficient scales follow a generalized
Pareto distribution given the layer scale and tail shape. In the unit
geometry, use the unit's scalar \eqn{\lambda_{kl}}{lambda[k,l]} for every
coordinate in that unit.
}
\examples{
x <- data.frame(z = seq(-1, 1, length.out = 15))
y <- x$z^2
fit <- gltbnn_fit(x, y, hidden = 3L, prior = "GLT",
                  control = list(init_steps = 10L, outer = 2L, weight_steps = 2L))
weights <- gltbnn_shrinkage(fit)
weights[, c("unit", "coordinate", "is_bias", "geometric_prior_sd")]
unique(weights[c("layer", "geometric_layer_scale", "geometric_tail_shape")])
}
\seealso{\code{\link{gltbnn_fit}}}
