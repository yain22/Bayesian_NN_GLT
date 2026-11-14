# Network nuisance fits

Prepare the household analysis file and five fold assignments with the accompanying data-preparation code. From this directory, run

~~~sh
Rscript --vanilla run.R 4
~~~

The final argument is the number of independent single-thread workers. Base R 4.5.1 and the bundled regression and classification engine were used for the reported fits. The same fold assignments and row identifiers must be used by the scoring script.

For each fold and architecture the program trains three networks per method. A Bernoulli network predicts 401(k) eligibility, and separate Gaussian regressions in the eligible and ineligible training households predict net financial assets. Each training sample supplies its own input means and standard deviations. The outcome regressions center and scale the dollar response in their own training arms and return conditional means in dollars. Every held-out household receives a prediction from both arms.

The grid crosses depths 2, 5 and 10 with widths 10 and 25. The six methods share an unpenalized 600-update initialization and a common sequence of 600 subsequent minibatches within each fold, nuisance and architecture. Batches contain up to 128 observations from successive random permutations, retaining each short final batch. The learning rate is 0.003. For the three Bayesian fits, 60 cycles of ten proposals use the package's default settings for coefficient-level shrinkage and the benchmark minibatch likelihood scaling. The outcome regressions use the Jeffreys noise-variance prior. Each Bayesian prediction averages the conditional means or probabilities from 256 independent weight draws. Dropout masks 20% of hidden activations during training. The L1 coefficient is 0.001 on all weights and biases. No outcome, eligibility, or predictive score on a held-out fold enters tuning or stopping.

Each completed checkpoint records the training protocol, prediction vector and validation row identifiers. Rerunning the command resumes completed work with the same protocol. At completion the program writes one compressed CSV of out-of-fold predictions. The raw propensity column is the estimated probability of eligibility. The two outcome columns give the fitted means under ineligibility and eligibility, respectively. The scoring step applies the specified propensity bounds.
