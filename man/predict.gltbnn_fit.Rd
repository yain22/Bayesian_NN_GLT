\name{predict.gltbnn_fit}
\alias{predict.gltbnn_fit}
\title{Predict conditional means or future responses}
\description{
Use draws from a fitted network's variational posterior to predict at new
observations. Predictions are returned in the original response units.
}
\usage{
\method{predict}{gltbnn_fit}(object, newdata,
       type = c("mean", "mean_draws", "response_draws", "mean_interval",
                "prediction_interval"),
       draws = 256L, level = 0.95, seed = 1L, ...)
}
\arguments{
  \item{object}{A model from \code{\link{gltbnn_fit}}.}
  \item{newdata}{Numeric matrix or numeric data frame with one row per new
  observation. Supply the same predictors as in training. When both data sets
  have column names, predictors are matched by name. Otherwise supply them
  in training order.}
  \item{type}{\code{"mean"} averages draws of the conditional mean.
  \code{"mean_draws"} returns those draws. \code{"response_draws"} samples
  future responses including Gaussian observation noise. \code{"mean_interval"}
  and \code{"prediction_interval"} return equal-tailed pointwise intervals
  for the conditional mean and a future response, respectively.}
  \item{draws}{Positive integer number of posterior draws. At least two are
  required for intervals.}
  \item{level}{Probability within each pointwise interval, strictly between
  zero and one.}
  \item{seed}{Nonnegative integer random seed. The previous \code{.Random.seed}
  is restored after prediction.}
  \item{...}{Reserved for future use.}
}
\details{
In each draw the same sampled network evaluates all rows of \code{newdata}.
A mean draw is its network output. For a response draw, the method also
samples a noise variance from its variational factor and adds independent
Gaussian errors at the new observations, conditional on that variance.
Intervals use sample quantiles of the corresponding draws.
}
\value{
For \code{"mean"}, a numeric vector with one entry per observation.
For draws, a numeric matrix with one row per observation and one column per
draw. For intervals, a two-column matrix named \code{lower} and \code{upper}.
}
\seealso{\code{\link{gltbnn_fit}} for a fitting and prediction example.}
