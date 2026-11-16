# Regularization in frequentist and Bayesian neural networks

Suppose a hidden unit uses trunk diameter to predict timber volume. At a fixed architecture, all three frequentist neural networks (FNNs) and all three Bayesian neural networks (BNNs) use the same network function. How do they fit the diameter weight and use it to predict the volume of a new tree? [Explore the interactive comparison](https://yain22.github.io/Bayesian_NN_GLT/computation/regularization.html).

| Method | What happens during training? |
| --- | --- |
| Unpenalized FNN | Adam updates the weights and biases to reduce the training loss. The diameter weight incurs no explicit penalty. |
| Dropout FNN | A random mask sometimes removes a hidden unit’s activation. The other units must help predict volume on those training passes. |
| $L_1$ FNN | The loss includes $0.001\sum_a \lvert w_a\rvert$ over all weights and biases, including the output intercept. Increasing the magnitude of any one coefficient adds the same penalty per unit increase. |
| Global BNN (G) | All incoming weights and biases in a layer share a learned scale. The expected squares of all these coefficients inform how strongly the layer shrinks. |
| Global–local BNN (GL) | Each coefficient has its own half-Cauchy multiplier on the shared layer scale. The diameter weight can retain a large scale even when the layer scale is small. |
| Global–local–tail BNN (GLT) | Each coefficient scale has a generalized Pareto prior conditional on its layer’s scale and tail shape. A larger tail shape permits exceptionally large scales more often, and the fit learns this shape separately for each layer. |

## What the frequentist updates do

For regression, the unpenalized fit minimizes squared error. For class labels, it minimizes the negative log probability of the observed class. The dropout fit masks hidden activations independently with probability 0.2. A surviving activation is divided by 0.8, so its average over masks equals its unmasked value. At prediction time the network uses all units with their fitted weights. The $L_1$ fit adds $0.001\operatorname{sign}(w_a)$ to each loss gradient and passes the result to Adam. The penalty can favor exact zeros at an optimum. A finite Adam step can cross zero without landing on it, so a small fitted coefficient does not establish that the connection was removed. [Goodfellow, Bengio and Courville, *Deep Learning*, §§7.1.2 and 7.12](https://www.deeplearningbook.org/contents/regularization.html) · [Regression benchmark code](../../tools/replication/c2-additive.R)

## What the Bayesian updates do

Each Bayesian fit gives the diameter weight a Gaussian distribution with mean $m$ and standard deviation $s$. In the Gaussian regression model its conditional prior variance is $\sigma^2\lambda^2$, where $\sigma^2$ is the observation variance and $\lambda$ is the coefficient scale. The scale update uses the expected square $m^2+s^2$. In return, a larger expected prior precision strengthens the pull toward zero in the next weight update. The variational objective also includes the entropy of the weight distribution. For a Gaussian this contributes $\log s$ plus a constant, so shrinking the fitted spread toward zero costs objective value even at a fixed mean. [Blei, Kucukelbir and McAuliffe on variational inference](https://arxiv.org/abs/1601.00670)

Each FNN predicts from one fitted weight vector. For the held-out tree, each BNN draws 256 networks from its fitted weight distribution and averages their predicted volumes. For classification, it averages the probabilities *after* applying the logistic or softmax function to each sampled network. [Prediction implementation](../../R/predict.R)

Bishop and Bishop develop the connection between parameter penalties, priors and Bayesian prediction in §§2.6.2–2.6.3. Their treatments of generalized weight decay (§9.2.2) and dropout (§9.6.1) give the background for the two regularized FNN benchmarks. [*Deep Learning: Foundations and Concepts*](https://www.bishopbook.com/)
