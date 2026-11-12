library(gltbnn)

# Independent 38-decimal integrals in r=1/xi, 0<r<=2, including the 1/r
# Jacobian. At the first configuration the mode of eta lies strictly between
# the left boundary and the coarse grid's first interior point.
oracle <- rbind(
  c(30000,-291655,1.984382351862462,-.685301791335296,24762.82139931206,-4.266799667956),
  c(30000,-292500,1.998354803835609,-.692323933651665,26448.20735139996,-6.103264361790),
  c(100000,-1200000,1.999990604076632,-.693142482576191,297394.45748950759,-11.268377129646),
  c(650,-4400,1.998231789634048,-.692262293375115,1667.243076360272,-6.030050485906),
  c(130,-630,1.986382037513499,-.686291905148076,244.608174455161,-3.982771899461),
  c(40,-120,1.756305725213735,-.559691808514958,19.574137092905,-1.146464838367),
  c(200000,-2321986,1.970914960297021,-.678496902050782,161583.9854299633,-5.152483057577),
  c(4800000,4800000*(2*(log(2)-digamma(2))+1),
    .064727674733389,2.737566533134413,-17924778.611632257,-6.276261346730),
  c(2400000,-33843314,1.980914411998929,-.683558481003241,1970194.88745264,-6.396819712910),
  c(4800000,-71060787,1.996004060303720,-.691147171668361,4033967.224407587,-6.746221688714),
  c(24389000,24389000*(2*(log(2)-digamma(2))+1),
    .058597348125855,2.837065857643152,-93517057.70537420,-7.088476186208)
)
tail_factor <- getFromNamespace(".bnn_tail_factor","gltbnn")
for (i in seq_len(nrow(oracle))) {
  a <- oracle[i,]
  q <- tail_factor(a[1],a[2])
  stopifnot(q$r > 0, q$r <= 2, abs(q$r-a[3])<1e-10, abs(q$eta_mean-a[4])<1e-10,
            abs(q$log_normalizer-a[5])<max(1e-8,32*.Machine$double.eps*abs(a[5])),
            abs(q$entropy_eta-a[6])<1e-8)
}

# Check the concentrated legacy unit-scale factor with an independent
# standardized integral. Direct log-normalizer subtraction loses the entropy.
q <- 1e9
local <- getFromNamespace(".bnn_local_factor","gltbnn")(q,q,2L)
scale <- 1/sqrt(3*q)
cost <- function(z) {
  s <- scale*z
  # Taylor expansion for q*(exp(s)+exp(-2*s)/2-1.5), plus the d=2 tilt.
  s + q*s*s*(1.5 + s*(-.5 + s*(.375 + s*(-.125 + s*(33/720)))))
}
moment <- function(f) stats::integrate(function(z) f(z)*exp(-cost(z)), -14, 14,
                                         rel.tol=1e-11,abs.tol=1e-14)$value
Z <- moment(function(z) 1)
log_mean <- scale*moment(identity)/Z
ent <- log(scale*Z) + log_mean + moment(cost)/Z
stopifnot(abs(local$zeta-log_mean)<1e-13, abs(local$entropy-ent)<1e-9,
          abs(local$u-moment(function(z) exp(scale*z))/Z)<1e-12,
          abs(local$phi-moment(function(z) exp(-2*scale*z))/Z)<1e-12)

# For very small equal A=B=t, Z = 3/2*(-log(t)-EulerGamma)+log(2)/2+o(1)
# and t*E[lambda]*Z tends to 1. The two softened ends of the log-density
# contribute the exponential-integral constants in this expansion.
t <- 1e-307
base <- getFromNamespace(".bnn_scalar_locals","gltbnn")(t,t)
Z <- 1.5*(-log(t)+digamma(1)) + log(2)/2
stopifnot(abs(base$log_normalizer-log(Z))<1e-12,abs((t*base$u)*Z-1)<1e-11,
          abs(base$phi/base$u-1)<1e-12)

# For d=2 the variable sqrt(B)/lambda approaches a standard half-normal
# as A*sqrt(B) approaches zero. The numerator of E[lambda] is the d=1
# normalizer and the denominator approaches sqrt(pi/(2B)).
t <- 1e-70
legacy <- getFromNamespace(".bnn_local_factor","gltbnn")(t,t,2L)
Z <- 1.5*(-log(t)+digamma(1)) + log(2)/2
stopifnot(abs(legacy$u/(Z*sqrt(2*t/pi)) - 1)<1e-10,abs(t*legacy$phi-1)<1e-10,
          abs(legacy$zeta - (log(t)-digamma(.5)-log(2))/2)<1e-10)

# A subnormal rate can overflow the unused first moment of the scalar
# numerator distribution. The required grouped moment remains finite.
t <- 1e-312
Q <- (2*log(t))/3
Z <- 1.5*(-Q+digamma(1))+log(2)/2
legacy <- getFromNamespace(".bnn_local_factor","gltbnn")(t,1,2L)
stopifnot(abs(legacy$u-Z*sqrt(2/pi))<1e-9)

# A 256-dimensional legacy group just below the concentrated-rule switch.
# Reuse the independent polynomial energy, now with the d-1 linear tilt.
q <- 648806400
scale <- 1/sqrt(3*q)
cost <- function(z) {
  s <- scale*z
  255*s + q*s*s*(1.5 + s*(-.5 + s*(.375 + s*(-.125 + s*(33/720)))))
}
Z <- moment(function(z) 1)
log_mean <- scale*moment(identity)/Z
actual <- getFromNamespace(".bnn_local_factor","gltbnn")(q,q,256L)
stopifnot(abs(actual$entropy - (log(scale*Z)+log_mean+moment(cost)/Z)) < 1e-9,
          abs(actual$u - moment(function(z) exp(scale*z))/Z) < 1e-12,
          abs(actual$phi - moment(function(z) exp(-2*scale*z))/Z) < 1e-12)
