<img src="assets/mark.svg" align="right" width="150" alt="BNN GLT project mark">

# Bayesian neural network shrinkage explorer

An interactive companion to *Variational Learning of Bayesian Neural Networks with a family of global–local shrinkage priors*. Follow one hidden neuron in a regression network with three inputs, two ReLU layers of four and eight units, and one affine output.

The four tabs let you select a neuron and trace its incoming connections, explore the spread of its weight prior, compare shrinkage scales across a layer, and see how a pruning decision changes the architecture. The examples run entirely in the browser. They illustrate prior distributions and model structure and perform no posterior fitting.

## Open the illustration

[Open the interactive illustration](https://yain22.github.io/Bayesian_NN_GLT/)

Open [index.html](index.html) from a downloaded copy of the repository. The hosted version is available through the repository’s GitHub Pages deployment. The static files require no build or dependencies. To serve a local copy, run the following command from the folder containing the entry page and open [the local preview](http://localhost:8000).

~~~sh
python -m http.server 8000 --bind 127.0.0.1
~~~

## Read the network

Selecting a neuron highlights every coefficient entering it, including the bias. A selected first-layer neuron has three incoming weights and a bias. A second-layer neuron has four incoming weights and a bias. The coefficients in each group can have different values while sharing a conditional prior standard deviation, $\sigma\lambda_{k\ell}$, for unit $k$ in layer $\ell$. In the examples $\sigma=1$, so a scale of $0.3$ gives each coefficient in that group a zero-centered Gaussian prior with standard deviation $0.3$.

| Prior | How the hidden units share scales |
| --- | --- |
| Global (G) | Every unit in layer $\ell$ uses the same scale, $\lambda_{k\ell}=\tau_\ell$. |
| Global–local (GL) | A half-Cauchy multiplier is specific to each unit, $\lambda_{k\ell}=\tau_\ell r_{k\ell}$ with $r_{k\ell}\sim C^+(0,1)$. |
| Global–local–tail (GLT) | The unit scales follow a generalized Pareto distribution with layer scale $\tau_\ell$ and tail shape $\xi_\ell$. The paper estimates one tail shape per layer. |

The tail plot asks how often a unit scale exceeds a multiple of the layer scale. For example, $P(\lambda_{k\ell}>10\tau_\ell)$ is about 6.3% under GL and 9.1% under GLT at $\xi_\ell=1$. These are probabilities under the specified priors. Predictive accuracy and interval coverage require evaluating fitted networks on data.

Ghosh, Yao, and Doshi-Velez also group the incoming weights of a hidden unit. After fitting their horseshoe networks, they can prune a unit when its fitted scale has sufficiently high probability of lying below a chosen threshold. The paper illustrated here retains every hidden unit and compares predictions across G, GL, and GLT.

## References

- Ghosh, Yao, and Doshi-Velez (2018). [Structured Variational Learning of Bayesian Neural Networks with Horseshoe Priors](https://proceedings.mlr.press/v80/ghosh18a.html).
- Ghosh, Yao, and Doshi-Velez (2019). [Model Selection in Bayesian Neural Networks via Horseshoe Priors](https://jmlr.org/papers/v20/19-236.html).
- Lee, Zhao, Pati, and Mallick (2024). [Tail-adaptive Bayesian shrinkage](https://doi.org/10.1214/24-EJS2317).
