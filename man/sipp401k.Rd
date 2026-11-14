\name{sipp401k}
\alias{sipp401k}
\docType{data}
\title{401(k) eligibility and household net financial assets}
\description{
Household observations from wave 4 of the 1990 U.S. Survey of Income and
Program Participation, collected in 1991. The sample comprises households
whose reference person was aged 25 to 64, with at least one employed member
and no self-employed members.
}
\usage{sipp401k}
\format{
A data frame with 9,915 rows and 11 columns. Each row represents one
household. Indicator variables take the value 1 when the condition holds
and 0 otherwise.
\describe{
  \item{e401}{Indicator of household eligibility for an employer-sponsored 401(k) plan.}
  \item{net_tfa}{Household net financial assets in 1991 U.S. dollars.}
  \item{age}{Age of the household head in years.}
  \item{inc}{Annual household income in 1991 U.S. dollars.}
  \item{educ}{Years of education of the household head.}
  \item{fsize}{Number of household members.}
  \item{marr}{Indicator of married status.}
  \item{twoearn}{Indicator of a two-earner household.}
  \item{db}{Indicator of defined-benefit pension coverage.}
  \item{pira}{Indicator of individual retirement account participation.}
  \item{hown}{Indicator of home ownership.}
}}
\source{
U.S. Census Bureau, Survey of Income and Program Participation, 1990 panel,
wave 4. The package contains the 11 analysis variables from the 401(k)
replication data of Achim Ahrens, Victor Chernozhukov, Christian Hansen,
Damian Kozbur, Mark Schaffer and Thomas Wiemann (2026), Harvard Dataverse,
\doi{10.7910/DVN/T81OHQ}. The extract retains the original household order
and is distributed under
\href{https://creativecommons.org/licenses/by/4.0/}{CC BY 4.0}.
}
\references{
Chernozhukov, V. and Hansen, C. (2004).
The effects of 401(k) participation on the wealth distribution.
\emph{The Review of Economics and Statistics}, 86, 735--751.
\doi{10.1162/0034653041811734}.

Chernozhukov, V. et al. (2018).
Double/debiased machine learning for treatment and structural parameters.
\emph{The Econometrics Journal}, 21, C1--C68. \doi{10.1111/ectj.12097}.
}
\examples{
with(sipp401k, table(e401))
}
\keyword{datasets}
