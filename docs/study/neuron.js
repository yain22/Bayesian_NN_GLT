"use strict";

// One fixed C2 replicate, independent of the across-replicate study selectors.
(() => {
  const status = document.getElementById("neuron-context");
  if (typeof c2NeuronScales === "undefined") {
    status.textContent = "The fitted scales could not be loaded. The C2 replication script below reproduces them.";
    return;
  }
  const controls = document.getElementById("neuron-controls");
  const layer = document.getElementById("neuron-layer");
  const unit = document.getElementById("neuron-unit");
  const inputs = document.getElementById("neuron-inputs");
  const table = document.getElementById("neuron-table");
  const body = document.getElementById("neuron-coefficients");
  const methods = ["G", "GL", "GLT"];
  const pageSize = 10;
  const number = new Intl.NumberFormat("en", { maximumSignificantDigits: 4 });

  function indexed(symbol, index) {
    const span = document.createElement("span");
    span.className = "coef-symbol";
    const sub = document.createElement("sub");
    sub.textContent = index;
    span.append(symbol, sub);
    return span;
  }
  function showNeuron() {
    const l = Number(layer.value);
    const k = Number(unit.value);
    const values = c2NeuronScales[l-1][k-1];
    const d = values.G.length;
    const first = Number(inputs.value);
    const last = Math.min(first + pageSize - 1, d-1);
    const jList = Array.from({length: last-first+1}, (_, i) => first+i);
    jList.push(d);
    const maximum = Math.max(...methods.flatMap(method => values[method]));
    const rows = jList.map(j => {
      const bias = j === d;
      const row = document.createElement("tr");
      const label = document.createElement("th");
      label.scope = "row";
      label.append(indexed(bias ? "b" : "w", bias ? String(k)+l : String(k)+l+","+j));
      const source = document.createElement("span");
      source.className = "coef-source";
      if (bias) {
        row.className = "neuron-bias";
        source.textContent = "Bias · j = " + d;
      } else if (l === 1) {
        source.append(indexed("x", String(j)), j <= 10 ? " · active predictor" : " · inactive predictor");
      } else {
        source.append(indexed("h", String(j)+"1"), " · layer 1 activation");
      }
      label.append(source);
      row.append(label);
      for (const method of methods) {
        const value = values[method][j-1];
        const cell = document.createElement("td");
        cell.append(number.format(value));
        const track = document.createElement("span");
        track.className = "coef-track";
        track.setAttribute("aria-hidden", "true");
        const bar = document.createElement("span");
        bar.style.width = (100*value/maximum)+"%";
        track.append(bar);
        cell.append(track);
        row.append(cell);
      }
      return row;
    });
    body.replaceChildren(...rows);
    status.textContent = "Layer " + l + ", neuron " + k + ". Showing coordinates " + first +
      "–" + last + " and bias " + d + ".";
    document.getElementById("neuron-caption").replaceChildren("Conditional prior scale ",
      indexed("δ", String(k)+l+",j"), ". Bars span 0 to "+number.format(maximum)+
      ", the largest scale across this neuron’s incoming coordinates and the three fits.");
  }
  function showLayer() {
    const units = c2NeuronScales[Number(layer.value)-1];
    const selected = unit.value || "1";
    unit.replaceChildren(...units.map((_, i) => new Option(String(i+1), String(i+1))));
    unit.value = selected;
    const count = units[0].G.length - 1;
    const ranges = [];
    for (let first = 1; first <= count; first += pageSize) {
      const last = Math.min(first+pageSize-1, count);
      ranges.push(new Option(first+"–"+last, String(first)));
    }
    inputs.replaceChildren(...ranges);
    showNeuron();
  }
  layer.addEventListener("change", showLayer);
  unit.addEventListener("change", showNeuron);
  inputs.addEventListener("change", showNeuron);
  showLayer();
  controls.hidden = false;
  table.hidden = false;
})();
