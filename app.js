/* Browser-only exploration; scale examples use fixed quantiles, not fitted weights. */
(() => {
  "use strict";

  const byId = id => document.getElementById(id);
  const tabs = [...document.querySelectorAll(".tab")];
  const priorButtons = [...document.querySelectorAll("[data-prior]")];
  const methodButtons = [...document.querySelectorAll("[data-method]")];
  const layers = {
    1: {tau: 0.5, xi: 1, quantiles: [0.25, 0.6, 0.4, 0.73]},
    2: {tau: 0.5, xi: 1, quantiles: [0.35, 0.58, 0.24, 0.72, 0.60, 0.48, 0.31, 0.68]}
  };
  const state = {layer: 1, unit: 2, spread: 0.6, prior: "GL", method: "ghosh", chance: 95, topic: "spread"};
  const priorText = {
    G: "G · Every neuron in this layer has the same prior spread.",
    GL: "GL · A half-Cauchy multiplier lets one neuron have a larger or smaller scale than its neighbors.",
    GLT: "GLT · The unit scales follow a generalized Pareto distribution. This example fixes the tail shape at 1. Explore other shapes in the Tail shape tab."
  };

  function updateNetwork() {
    const pruned = state.topic === "prune" && state.method === "ghosh" && state.chance > 90;
    const layer = layers[state.layer];
    let priorSd;
    if (state.topic === "spread") priorSd = state.spread;
    if (state.topic === "scales") priorSd = selectedScale(layer.quantiles[state.unit - 1],layer.tau);
    window.drawNetwork(byId("network"), {layer:state.layer, unit:state.unit, pruned, priorSd}, selectNeuron);
    byId("selection-name").textContent = "Layer " + state.layer + " · Neuron " + state.unit;
    byId("selection-detail").textContent = pruned
      ? "The dashed outline marks its former position. Its incoming and outgoing connections have been removed."
      : (state.layer === 1 ? "Three input weights" : "Four incoming weights") +
        " and one bias share a prior spread. The square marked 1 supplies the bias input.";
  }

  function selectNeuron(layer, unit) {
    state.layer = layer;
    state.unit = unit;
    render();
  }

  function updateSpread() {
    const sd = state.spread.toFixed(2);
    byId("spread-value").value = sd;
    window.PriorPlots.bell(byId("bell-chart"),state.spread);
    byId("spread-readout").textContent =
      "About 68% of the prior lies between −" + sd + " and +" + sd +
      ". A narrower bell puts more weight near zero.";
  }

  function selectedScale(q, tau) {
    if (state.prior === "G") return tau;
    if (state.prior === "GL") return tau * Math.tan(Math.PI * q / 2);
    // At xi = 1 the GPD quantile is tau * q / (1 - q).
    return tau * q / (1 - q);
  }

  function span(className, content) {
    const item = document.createElement("span");
    item.className = className;
    if (content !== undefined) item.textContent = content;
    return item;
  }

  function updateScales() {
    const layer = layers[state.layer];
    byId("prior-explainer").textContent = priorText[state.prior];
    byId("layer-scale").value = layer.tau;
    byId("layer-scale-value").value = layer.tau.toFixed(2);
    byId("percentile-control").hidden = state.prior === "G";
    const percentile = Math.round(layer.quantiles[state.unit - 1] * 100);
    byId("percentile").value = percentile;
    byId("percentile-value").value = percentile + "%";
    priorButtons.forEach(button => button.setAttribute("aria-pressed", button.dataset.prior === state.prior));
    const bars = byId("unit-bars");
    const focused = bars.contains(document.activeElement);
    bars.replaceChildren(...layer.quantiles.map((q,index) => {
      const sd = selectedScale(q,layer.tau);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "scale-row";
      button.setAttribute("aria-pressed",state.unit === index + 1);
      button.setAttribute("aria-label","Neuron " + (index + 1) + ", prior standard deviation " + sd.toFixed(2));
      const track = span("scale-track");
      const fill = span("scale-fill");
      fill.style.width = (100 * Math.min(1,sd / 5)) + "%";
      if (state.layer === 2) fill.style.backgroundColor = "var(--rust)";
      track.append(fill);
      button.append(span("scale-name","Neuron " + (index + 1)),track,span("scale-value",sd.toFixed(2)));
      button.addEventListener("click",() => selectNeuron(state.layer,index + 1));
      return button;
    }));
    if (focused) bars.children[state.unit - 1].focus({preventScroll:true});
  }

  function updateTail() {
    const xi = layers[state.layer].xi;
    byId("tail-shape").value = xi;
    byId("tail-label").textContent = "GLT tail shape ξ in layer " + state.layer;
    byId("tail-value").value = xi.toFixed(2);
    window.PriorPlots.tail(byId("tail-chart"),xi);
    const probs = window.PriorPlots.tailProbs(xi);
    for (const key of ["g","gl","glt"]) {
      byId("tail-" + key).textContent = (100 * probs[key]).toFixed(key === "g" ? 0 : 1) + "%";
    }
  }

  function updatePruning() {
    methodButtons.forEach(button => button.setAttribute("aria-pressed",button.dataset.method === state.method));
    byId("prune-value").value = state.chance + "%";
    const unit = "Neuron " + state.unit + " in layer " + state.layer;
    if (state.method === "ours") {
      byId("prune-result").textContent = unit + " stays. We retain all four and eight hidden units when comparing predictive accuracy.";
    } else if (state.chance > 90) {
      const left = layers[state.layer].quantiles.length - 1;
      byId("prune-result").textContent = unit + " is pruned in this example. This leaves " + left + " units in that layer.";
    } else {
      byId("prune-result").textContent = unit + " stays. The example probability has not crossed the 90% cutoff.";
    }
  }

  const updatePane = {spread:updateSpread,scales:updateScales,tail:updateTail,prune:updatePruning};
  function render() {
    updateNetwork();
    updatePane[state.topic]();
  }

  function showTopic(topic) {
    state.topic = topic;
    tabs.forEach(tab => {
      const active = tab.id === "tab-" + topic;
      tab.setAttribute("aria-selected",active);
      tab.tabIndex = active ? 0 : -1;
      byId(tab.getAttribute("aria-controls")).hidden = !active;
    });
    render();
  }

  tabs.forEach((tab,i) => {
    tab.addEventListener("click",() => showTopic(tab.id.slice(4)));
    tab.addEventListener("keydown",event => {
      const next = {ArrowRight:(i + 1) % tabs.length, ArrowLeft:(i + tabs.length - 1) % tabs.length,
                    Home:0, End:tabs.length - 1}[event.key];
      if (next === undefined) return;
      event.preventDefault();
      tabs[next].focus();
      showTopic(tabs[next].id.slice(4));
    });
  });
  priorButtons.forEach(button => button.addEventListener("click",() => {
    state.prior = button.dataset.prior;
    render();
  }));
  methodButtons.forEach(button => button.addEventListener("click",() => {
    state.method = button.dataset.method;
    render();
  }));
  byId("spread").addEventListener("input",event => {
    state.spread = event.target.valueAsNumber;
    render();
  });
  byId("layer-scale").addEventListener("input",event => {
    layers[state.layer].tau = event.target.valueAsNumber;
    render();
  });
  byId("percentile").addEventListener("input",event => {
    layers[state.layer].quantiles[state.unit - 1] = event.target.valueAsNumber / 100;
    render();
  });
  byId("tail-shape").addEventListener("input",event => {
    layers[state.layer].xi = event.target.valueAsNumber;
    updateTail();
  });
  byId("prune-chance").addEventListener("input",event => {
    state.chance = event.target.valueAsNumber;
    render();
  });
  render();
})();
