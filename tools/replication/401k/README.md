# Five-fold AIPW analysis of 401(k) eligibility

This supplement replication estimates the average effect of eligibility for a 401(k) plan on household net financial assets in 1991 dollars. The sample and nine adjustment variables follow Chernozhukov et al. (2018). The six network methods and six architectures use the same five treatment-stratified folds. The treatment indicator models eligibility, and separate outcome regressions model assets in the two eligibility groups.

## Run the GLT example in Supplement Section S7.4

From the extracted replication directory, run

~~~sh
Rscript --vanilla example_401k.R
~~~

The example reads the included household data and five fold assignments. For each held-out fold it fits separate GLT networks for the two outcome regressions and the eligibility probability, using ten hidden layers of 25 units. It prints the AIPW estimate, standard error and 95% confidence limits in 1991 dollars. Rounded to the nearest dollar, these are 8,080, 1,135, 5,855 and 10,305, respectively.

## Reproduce the full comparison

Use R with its recommended package **foreign**. The reported results were computed with R 4.5.1. Python 3 with NumPy is used for independent verification. From the extracted archive directory, run these commands in order.

~~~sh
Rscript --vanilla data/prepare_401k.R data
cd fit
Rscript --vanilla run.R 4
cd ..
Rscript --vanilla score/score_aipw.R data/analysis.csv fit/predictions.csv.gz reproduced
python score/verify_aipw.py data/analysis.csv fit/predictions.csv.gz reproduced
~~~

The preparation script verifies the included public sample and recreates the analysis file and fold assignments. If the original sample is missing, the script downloads it from a fixed revision of the original replication repository. The fitting command starts four workers. Set its final argument to the number of workers available. The bundled R code implements the 540 nuisance fits and saves checkpoints so the run can resume. Details of scaling, initialization, batching and prediction appear in the fitting notes.

Table S6 reports the five-fold AIPW estimates and their standard errors. The **reference** directory contains the estimates and additional replication outputs. The estimates file reports the mean influence-function score, its standard error, the 95% interval and the counts of propensity scores clipped at each bound. The primary bounds are 0.01 and 0.99. The remaining rows give the sensitivity analyses. The optional diagnostics file reports the held-out regression errors, propensity Brier scores and propensity distributions. To compare the predictions against the published summaries directly, run the verifier again with **reference** as its last argument. Floating-point and numerical-library differences across platforms may affect individual network fits.

The source notes give the sample definition, source and file hashes. For the original analysis, see [Chernozhukov et al., *Double/debiased machine learning for treatment and structural parameters*](https://doi.org/10.1111/ectj.12097).
