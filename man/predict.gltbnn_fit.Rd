\name{predict.gltbnn_fit}
\alias{predict.gltbnn_fit}
\title{Predict responses or class probabilities}
\description{
Use draws from a fitted network's variational posterior to predict at new
observations. Regression predictions use the original response units.
Classification predictions retain the training labels and class order.
}
\usage{
\method{predict}{gltbnn_fit}(object, newdata,
       type = NULL,
       draws = 256L, level = 0.95, seed = 1L, ...)
}
\arguments{
  \item{object}{A model from \code{\link{gltbnn_fit}}.}
  \item{newdata}{Numeric matrix or numeric data frame with one row per new
  observation. Supply the same predictors as in training. When both data sets
  have column names, predictors are matched by name. Otherwise supply them
  in training order. Prediction reports an error if applying the training
  standardization or evaluating the predictions exceeds the finite numerical range.}
  \item{type}{For regression, the default \code{"mean"} averages draws of the conditional mean.
  \code{"mean_draws"} returns those draws. \code{"response_draws"} samples
  future responses including Gaussian observation noise. \code{"mean_interval"}
  and \code{"prediction_interval"} return equal-tailed pointwise intervals
  for the conditional mean and a future response, respectively.
  For classification, the default \code{"class"} gives the class with the
  largest posterior mean probability, breaking ties in training level order.
  Use \code{"prob"} for those mean probabilities and \code{"prob_draws"} for
  class probabilities under each sampled network.}
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
In regression, a mean draw is its network output. For a response draw, the method also
samples a noise variance from its variational factor and adds independent
Gaussian errors at the new observations, conditional on that variance.
Intervals use sample quantiles of the corresponding draws.

For classification, each network draw yields a vector of probabilities per
observation by applying the logistic or softmax link to its affine outputs.
The \code{"prob"} result averages the resulting probabilities over draws.
For binary outcomes the second level is the event modeled by the logit.
}
\value{
For \code{"mean"}, a numeric vector with one entry per observation.
For regression draws, a numeric matrix with one row per observation and one column per
draw. For intervals, a two-column matrix named \code{lower} and \code{upper}.
For \code{"class"}, a factor with the training levels. For \code{"prob"}, a
matrix with one row per observation and one named column per class. For
\code{"prob_draws"}, an array indexed by observation, class, and draw.
}
\seealso{\code{\link{gltbnn_fit}} for a fitting and prediction example.}
