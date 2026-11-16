"""Build the computation guide with native MathML and its downloadable R examples."""

from html import escape
from pathlib import Path

from latex2mathml.converter import convert


destination = Path(__file__).resolve().parents[2] / "docs" / "computation"


def equation(tex):
    markup = convert(tex, display="block").replace("<math ", f'<math alttext="{escape(tex, quote=True)}" ', 1)
    return f'<div class="formula">{markup}</div>'


def model_section(prior, title, explanation, formulas, interpretation):
    code = (destination / f"{prior}.R").read_text(encoding="utf-8")
    key = prior.lower()
    return f"""<section id="{key}" class="section-grid" aria-labelledby="{key}-title">
  <div>
    <h2 id="{key}-title">{prior}-BNN</h2>
    <h3>{title}</h3>
    <p>{explanation}</p>
    {''.join(equation(formula) for formula in formulas)}
    <p>{interpretation}</p>
  </div>
  <div class="codebox">
    <div class="codebar"><span>{prior}.R</span><a href="{prior}.R" download>Download R script</a></div>
    <pre tabindex="0" aria-label="{prior}-BNN example"><code id="{key}-code">{escape(code, quote=False)}</code></pre>
  </div>
</section>"""


sections = [
    model_section(
        "G", "One scale per hidden layer",
        "The layer size fixes the half-Cauchy prior scale at 1/<i>M</i><sub>l</sub>, where "
        "<i>M</i><sub>l</sub> counts the hidden weights and biases in layer <i>l</i>. "
        "The fitted layer scale is random. Its squared-scale update uses the expected square of every coefficient. "
        "Their next weight updates use the same scale precision.",
        [r"\lambda_{kl,j}=\tau_l,\qquad P_{kl,j}=\mathbb{E}_q(\tau_l^{-2}).",
         r"\chi_{\tau,l}\sim\operatorname{IG}(1/2,M_l^2),\qquad \tau_l^2\mid\chi_{\tau,l}\sim\operatorname{IG}(1/2,1/\chi_{\tau,l})."],
        "The example returns one row per layer. <code>geometric_layer_scale</code> is the "
        "geometric mean of the fitted layer scale, and <code>scale_precision</code> is its "
        "expected inverse square.",
    ),
    model_section(
        "GL", "One local multiplier per coefficient",
        "Each hidden weight and bias has its own half-Cauchy multiplier. A fitting cycle "
        "updates each squared multiplier and its auxiliary from that coefficient’s expected square, "
        "then updates the shared layer scale and its auxiliary. The layer scale and individual multipliers "
        "have independent variational factors. Their products share the same fitted layer scale, so the "
        "derived coefficient scales in a layer can be dependent under <i>q</i>. The fitted weights and "
        "biases have independent Gaussian factors.",
        [r"\lambda_{kl,j}=\tau_l r_{kl,j},\qquad r_{kl,j}\sim C^+(0,1).",
         r"P_{kl,j}=\mathbb{E}_q(\tau_l^{-2})\,\mathbb{E}_q(r_{kl,j}^{-2}).",
         r"\chi_{r,kl,j}\sim\operatorname{IG}(1/2,1),\qquad r_{kl,j}^2\mid\chi_{r,kl,j}\sim\operatorname{IG}(1/2,1/\chi_{r,kl,j})."],
        "For the first hidden unit the example displays the diameter weight, height weight and bias "
        "in that order. Each has its own scale precision. <code>geometric_prior_sd</code> "
        "is the geometric mean of its conditional prior standard deviation on the standardized scale.",
    ),
    model_section(
        "GLT", "A learned tail shape in each layer",
        "Each scalar coefficient scale follows a generalized Pareto law given the layer scale "
        "and tail shape. In the fitted approximation each coefficient scale has its own independent "
        "variational factor, separate from the layer-scale and tail-shape factors. "
        "A fitting cycle updates the individual scales, the layer scale, the "
        "coefficient auxiliaries and finally the tail shape. One-dimensional quadrature supplies "
        "the local-scale and tail-shape moments.",
        [r"\lambda_{kl,j}\mid\tau_l,\xi_l\sim\operatorname{GPD}(\tau_l,\xi_l),",
         r"P_{kl,j}=\mathbb{E}_q(\lambda_{kl,j}^{-2})."],
        "The first summary gives the geometric means of the layer scale and tail shape. "
        "The second shows the precision of each incoming weight and bias of the first hidden unit.",
    ),
]

page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Variational fitting and runnable R examples for global, global–local and global–local–tail Bayesian neural networks.">
  <title>Fitting G, GL and GLT networks · gltbnn</title>
  <link rel="icon" href="../assets/mark.svg" type="image/svg+xml">
  <link rel="stylesheet" href="../package.css">
  <style>
    .guide-intro { padding-block:clamp(2rem,5vw,4rem) 1.75rem; }
    .guide-intro h1 { max-width:17ch; }
    .guide-intro p, .fit-intro p { max-width:72ch; color:var(--muted); }
    .guide-routes { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border-block:1px solid var(--rule); }
    .guide-routes a { display:block; padding:1.25rem; text-decoration:none; color:var(--ink); }
    .guide-routes a + a { border-left:1px solid var(--rule); }
    .guide-routes a:first-child { background:#e8f1ef; }
    .guide-routes a:hover { background:white; }
    .guide-routes strong { display:block; font:600 1.3rem/1.25 var(--serif); }
    .guide-routes span { display:block; margin-top:.45rem; color:var(--muted); font-size:.93rem; }
    .fit-intro { padding:2.5rem 0 1.5rem; }
    .fit-intro details { margin:1rem 0; }
    .fit-intro summary { cursor:pointer; color:var(--navy); font-weight:600; }
    .fit-intro pre { max-width:44rem; background:#efece5; }
    .section-grid { grid-template-columns:minmax(0,.95fr) minmax(0,1.3fr); gap:clamp(1.25rem,4vw,3.5rem); }
    .section-grid h3 { font:600 1.07rem/1.5 var(--sans); letter-spacing:0; }
    .section-grid > div { min-width:0; }
    .formula { overflow-x:auto; margin:1rem 0; }
    math[display=block] { min-width:max-content; font-size:1.05rem; }
    .codebar a { color:#fff; }
    @media (max-width:860px) { .section-grid { grid-template-columns:minmax(0,1fr); } }
    @media (min-width:591px) and (max-width:940px) {
      .guide-routes { grid-template-columns:repeat(2,minmax(0,1fr)); }
      .guide-routes a:nth-child(3) { border-left:0; }
      .guide-routes a:nth-child(n+3) { border-top:1px solid var(--rule); }
    }
    @media (max-width:590px) { .guide-routes { grid-template-columns:1fr; } .guide-routes a + a { border-left:0; border-top:1px solid var(--rule); } }
  </style>
</head>
<body>
<a class="skip" href="#content">Skip to content</a>
<header class="topbar"><div class="wrap bar-inner">
  <a class="wordmark" href="../" aria-label="gltbnn home">glt<span>bnn</span></a>
  <nav aria-label="Computational methods"><a href="../">Package home</a><a href="#examples">R examples</a><a href="#checks">Numerical checks</a><a href="../reference.pdf">R reference</a></nav>
</div></header>
<main id="content" class="wrap">
<header class="guide-intro">
  <h1>How the networks are fitted</h1>
  <p class="intro">To predict timber volume, a network learns how strongly to weight trunk diameter and height. Each Bayesian fitting cycle updates the weights, uses their expected squares to learn shrinkage scales, and returns a precision for the next weight update.</p>
</header>
<div class="guide-routes" role="navigation" aria-label="Explore the computation">
  <a href="cycle.html"><strong>Follow one fitting cycle →</strong><span>Trace the diameter weight in a real fit, from its Gaussian factor to its next prior precision.</span></a>
  <a href="regularization.html"><strong>Compare BNNs and FNNs →</strong><span>See how all six methods fit a weight and predict for a new observation.</span></a>
  <a href="#examples"><strong>Fit the networks in R →</strong><span>Run the same timber example with a global, horseshoe or global–local–tail prior.</span></a>
  <a href="algorithms.html"><strong>Check an update equation →</strong><span>Compare the three algorithms with their R implementation and the supplement.</span></a>
</div>
<section id="examples" class="fit-intro" aria-labelledby="examples-title">
  <h2 id="examples-title">Fit the timber example in R</h2>
  <p>All three examples predict timber volume from trunk diameter (the <code>Girth</code> column) and height in <code>datasets::trees</code>. They share the training split, starting weights and two hidden layers. Every fifth tree is held out for prediction. <a href="../#install">Install the package</a>, then run the <a href="#g">G</a>, <a href="#gl">GL</a> or <a href="#glt">GLT</a> script.</p>
  <details><summary>Fitting budget and notation</summary>
  <p>The examples use 600 initialization steps, 60 fitting cycles and the Jeffreys prior on the standardized observation variance. They return predictions in cubic feet. The scale precision <i>P</i><sub>kl,j</sub> in the equations below is 𝔼<sub>q</sub>[λ<sub>kl,j</sub><sup>−2</sup>]. Multiplying by 𝔼<sub>q</sub>[σ<sup>−2</sup>] gives the effective precision for a hidden coefficient in Gaussian regression.</p>
  <p>To fit a GLT network for 180 cycles, supply the predictor matrix <code>x</code> and response <code>y</code>, and set the cycle budget in <code>control</code>.</p>
  <pre><code>gltbnn_fit(x, y, prior = "GLT",
           control = list(outer = 180L, weight_steps = 10L))</code></pre>
  <p>Increasing <code>outer</code> extends the run while keeping ten successive weight-update attempts per cycle. The variance and shrinkage factors are still updated once per cycle.</p>
  <p>In the supplement, Section S1 derives the shared variational updates. Section S2 gives the G-BNN algorithm, S3 the GL-BNN algorithm and S4 the GLT-BNN algorithm.</p>
  <p>These half-Cauchy expansions use random auxiliaries χ<sub>τ,l</sub> for the layer and χ<sub>r,kl,j</sub> for the GL coefficient. In a fitted layer’s R state, <code>$aux</code> stores the G or GL layer auxiliary factor and <code>$nu</code> stores the GL coefficient auxiliary factors. Under GLT, <code>$nu</code> stores the generalized Pareto mixing factors. Within an inverse-gamma factor, <code>a</code> is its shape parameter.</p>
  </details>
</section>
""" + "\n".join(sections) + """
<section id="checks" class="section-grid" aria-labelledby="checks-title">
  <div><h2 id="checks-title">Numerical checks</h2></div>
  <div>
    <p>Supplement Section S5.1 compares the GLT coefficient-scale quadrature with independent adaptive integration for nine pairs of local-factor parameters. <a href="check-quadrature.R" download>Download the R quadrature check</a>.</p>
    <p>Figure S11 follows ten initializations for each of the three priors in nine settings. Within each setting, the networks use the same training data, and all three priors use the same starting weights for each initialization. The 270 trajectories record the monitored ELBO at initialization and after each of 60 fitting cycles. <a href="elbo-replay.html">Replay the recorded fitting cycles</a>, <a href="elbo-initializations.pdf">view Figure S11</a>, or <a href="elbo-initializations.zip" download>download the traces, synthetic training inputs, and code to reproduce the figure or refit the networks</a>.</p>
  </div>
</section>
</main>
</body>
</html>
"""
(destination / "index.html").write_bytes(page.encode("utf-8"))
