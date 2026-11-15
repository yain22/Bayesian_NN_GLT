# Five-fold AIPW analysis of 401(k) eligibility

This analysis reproduces the main paper's 401(k) eligibility results. It estimates the adjusted difference in household net financial assets between households eligible and ineligible for a 401(k) plan, in 1991 dollars. The sample and nine adjustment variables follow Chernozhukov et al. (2018). The six network methods and six architectures use the same five treatment-stratified folds. The treatment indicator models eligibility, and separate outcome regressions model assets in the two eligibility groups.

## Run the GLT example in Supplement Section S7.3

The standalone script uses the public functions and survey data in **gltbnn 0.2.1**. Extract `causal_401k_replication.zip`. Its top-level folder contains both `example_401k.R` and the offline installer, `gltbnn_0.2.1.tar.gz`. If the package is needed, start R in that folder and run:

~~~r
install.packages("gltbnn_0.2.1.tar.gz", repos = NULL, type = "source")
~~~

With the package installed, run the following shell command from the folder containing `example_401k.R`:

~~~sh
Rscript --vanilla example_401k.R
~~~

To run the installed demo from any working directory, use this command in R:

~~~r
demo("aipw401k", package = "gltbnn", ask = FALSE)
~~~

The example builds five eligibility-stratified folds and calls `gltbnn_fit()` for two outcome regressions and one eligibility model in each fold. Each GLT network has ten hidden layers of 25 units. The script averages the household AIPW scores and prints the adjusted difference ($8,080), standard error ($1,135), and 95% confidence interval ($5,855 to $10,305), in 1991 dollars.

## Reproduce the full comparison

Use R with its recommended package **foreign**. The reported results were computed with R 4.5.1. The independent verifier requires Python 3 with NumPy. The last command assumes `python` starts that interpreter. If your installation uses `python3` or `py -3`, substitute that launcher for `python`. From the extracted archive directory, run these commands in order.

~~~sh
Rscript --vanilla data/prepare_401k.R data
cd fit
Rscript --vanilla run.R 4
cd ..
Rscript --vanilla score/score_aipw.R data/analysis.csv fit/predictions.csv.gz reproduced
python score/verify_aipw.py data/analysis.csv fit/predictions.csv.gz reproduced
~~~

The preparation script verifies the included public sample and recreates the analysis file and fold assignments. If the original sample is missing, the script downloads it from a fixed revision of the original replication repository. The fitting command starts four workers. Set its final argument to the number of workers available. The bundled R code implements the 540 nuisance fits and saves checkpoints so the run can resume. Details of scaling, initialization, batching and prediction appear in the fitting notes.

The 401(k) results table reports the five-fold AIPW estimates and their standard errors. A total-effect interpretation requires the identification assumptions in Supplement S7.1, including that eligibility did not change whether a household held an individual retirement account. The **reference** directory contains the estimates and additional replication outputs. The estimates file reports the mean influence-function score, its standard error, the 95% interval and the counts of propensity scores clipped at each bound. The primary bounds are 0.01 and 0.99. The remaining rows give the sensitivity analyses. The optional diagnostics file reports the held-out regression errors, propensity Brier scores and propensity distributions. To compare the predictions against the published summaries directly, run the verifier again with **reference** as its last argument. Floating-point and numerical-library differences across platforms may affect individual network fits.

The source notes give the sample definition, source and file hashes. For the original analysis, see [Chernozhukov et al., *Double/debiased machine learning for treatment and structural parameters*](https://doi.org/10.1111/ectj.12097).
