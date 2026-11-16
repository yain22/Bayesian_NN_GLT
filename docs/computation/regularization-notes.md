# Regularization in frequentist and Bayesian neural networks

The first hidden unit combines standardized trunk diameter, standardized height and a bias. In the paper's notation its weighted sum is $z_{11}(\mathbf x)=w_{11,1}x_1+w_{11,2}x_2+b_{11}$, and $h_{11}(\mathbf x)=\max(0,z_{11}(\mathbf x))$. The comparison follows the diameter coefficient $w=w_{11,1}$ through six fits of the same network architecture. [Explore the interactive comparison](https://yain22.github.io/Bayesian_NN_GLT/computation/regularization.html).

| Method | What happens during training? |
| --- | --- |
| Unpenalized FNN | Adam updates the weights and biases to reduce the training loss. The diameter weight incurs no explicit penalty. |
| Dropout FNN | A random mask sometimes removes a hidden unit’s activation. The other units must help predict volume on those training passes. |
| $L_1$ FNN | The loss includes $\eta_{\rm reg}\sum_a \lvert w_a\rvert$ over all weights and biases, including the output intercept. Here $\eta_{\rm reg}=0.001$. |
| Global BNN (G) | All incoming weights and biases in a layer share a learned scale. The expected squares of all these coefficients inform how strongly the layer shrinks. |
| Global–local BNN (GL) | Each coefficient has its own half-Cauchy multiplier on the shared layer scale. The diameter weight can retain a large scale even when the layer scale is small. |
| Global–local–tail BNN (GLT) | Each coefficient scale has a generalized Pareto prior conditional on its layer’s scale and tail shape. A larger tail shape permits exceptionally large scales more often, and the fit learns this shape separately for each layer. |

## What the frequentist updates do

The regression loss is one half of the mean squared error on the training scale. For class labels the data term is the mean negative log probability of the observed class. Dropout independently masks each hidden activation with probability 0.2 per training observation. A surviving activation is divided by 0.8, so its average over masks equals its unmasked value. At prediction the fitted network uses all hidden activations at their ordinary scale. The $L_1$ fit adds $\eta_{\rm reg}\operatorname{sign}(w_a)$ to the loss gradient for each coefficient, choosing the zero subgradient at $w_a=0$. An Adam step can cross zero without landing on it. [Goodfellow, Bengio and Courville, *Deep Learning*, §§7.1.2 and 7.12](https://www.deeplearningbook.org/contents/regularization.html) · [Regression benchmark code](../../tools/replication/c2-additive.R)

## What the Bayesian updates do

The fitted variational factor for the diameter weight is $q(w)=\mathcal N(m,s^2)$. In Gaussian regression the conditional prior variance of a hidden coefficient is $\sigma^2\lambda_{kl,j}^2$ for the fitted choice $c_l=1$. Here $\sigma^2$ is the observation variance on the standardized response scale. Supplement Eq. (S8) distinguishes the scale precision $P_{kl,j}=\mathbb E_q(\lambda_{kl,j}^{-2})$ from the weight precision $\omega_{kl,j}=\phi_\sigma P_{kl,j}$, where $\phi_\sigma=\mathbb E_q(\sigma^{-2})$. For the selected weight, write $P=P_{11,1}$ and $\omega=\omega_{11,1}$.

The scale update uses $H_{11,1}=\mathbb E_q(w^2)=m^2+s^2$. Holding the noise and scale factors fixed, the expected negative log prior contributes $\omega(m^2+s^2)/2$ up to a constant. Its derivative with respect to $m$ is $+\omega m$. Maximizing the ELBO uses the negative of that derivative for the log-prior term. The entropy of the Gaussian weight factor contributes $\log s$ plus a constant. [Blei, Kucukelbir and McAuliffe on variational inference](https://arxiv.org/abs/1601.00670)

For GLT, the local update has density proportional to $\lambda^{-1}\exp\{-A\lambda-B/(2\lambda^2)\}$ for $\lambda>0$. The implementation integrates $\mathbb E_q(\lambda)$ by quadrature and recovers $P=A\mathbb E_q(\lambda)/B$, as in Supplement Algorithm S3. Here $A=\mathbb E_q(\tau_1^{-1})\mathbb E_q(\nu_{11,1}^{-1})$ and $B=\phi_\sigma H_{11,1}$.

Each FNN evaluates $f_{\widehat{\mathcal W}}(\mathbf x_*)$ once. Each BNN averages $f_{\mathcal W^{(s)}}(\mathbf x_*)$ over 256 draws from $q_{\boldsymbol\theta}(\mathcal W)$. Here $\mathbf x_*$ is standardized using the training trees. To obtain cubic feet, multiply each network output by the training sample standard deviation of volume $s_y$ and add its training mean $\bar y$. The Bayesian draws represent conditional means at the sampled weights. For classification the network instead averages probabilities after applying the logistic or softmax function at each variational weight draw. [Prediction implementation](../../R/predict.R)

Bishop and Bishop develop the connection between parameter penalties, priors and Bayesian prediction in §§2.6.2–2.6.3. Their treatments of generalized weight decay (§9.2.2) and dropout (§9.6.1) give the background for the two regularized FNN benchmarks. [*Deep Learning: Foundations and Concepts*](https://www.bishopbook.com/)
