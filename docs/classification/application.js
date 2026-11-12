"use strict";

const element = id => document.getElementById(id);
const percent = value => (100 * value).toFixed(1) + "%";
const pageSize = 10;
const allIndices = mnist.truth.map((_, i) => i);
const correct = i => mnist.truth[i] === mnist.predicted[i];
let visibleIndices = allIndices;
let page = 0;
let selected = 0;

function renderPrediction() {
  const observed = mnist.truth[selected];
  const predicted = mnist.predicted[selected];
  element("selection").textContent =
    `Test image #${mnist.sourceIndex[selected]} · Recorded ${observed} · Predicted ${predicted}`;
  const rows = mnist.probabilities[selected].map((prob, digit) => {
    const row = document.createElement("div");
    row.className = digit === predicted ? "prob-row winner" : "prob-row";
    const label = document.createElement("span");
    label.textContent = String(digit);
    label.setAttribute("aria-label", "Digit " + digit);
    const track = document.createElement("span");
    track.className = "prob-track";
    track.setAttribute("aria-hidden", "true");
    const fill = document.createElement("span");
    fill.className = "prob-fill";
    fill.style.width = (100 * prob) + "%";
    track.append(fill);
    const value = document.createElement("span");
    value.className = "value";
    value.textContent = prob > 0 && prob < 0.0005 ? "<0.1%" : percent(prob);
    row.append(label, track, value);
    return row;
  });
  element("probabilities").replaceChildren(...rows);
  for (const button of element("gallery").children) {
    button.setAttribute("aria-pressed", String(Number(button.dataset.index) === selected));
  }
}

function renderGallery() {
  const start = page * pageSize;
  const indices = visibleIndices.slice(start, start + pageSize);
  const sheetRows = Math.ceil(mnist.truth.length / 10);
  const buttons = indices.map(i => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "digit";
    button.dataset.index = i;
    button.setAttribute("aria-label", `Test image ${mnist.sourceIndex[i]}, recorded digit ${mnist.truth[i]}`);
    const label = document.createElement("span");
    label.className = "digit-label";
    label.textContent = "Label " + mnist.truth[i];
    const glyph = document.createElement("span");
    glyph.className = "glyph";
    glyph.setAttribute("aria-hidden", "true");
    glyph.style.backgroundSize = `1000% ${100 * sheetRows}%`;
    glyph.style.backgroundPosition = `${100 * (i % 10) / 9}% ${100 * Math.floor(i / 10) / (sheetRows - 1)}%`;
    button.append(label, glyph);
    button.addEventListener("click", () => {
      selected = i;
      renderPrediction();
    });
    return button;
  });
  element("gallery").replaceChildren(...buttons);
  element("previous").disabled = page === 0;
  element("next").disabled = start + indices.length >= visibleIndices.length;
  element("page-count").textContent =
    `${start + 1}–${start + indices.length} of ${visibleIndices.length}`;
  selected = indices[0];
  renderPrediction();
}

element("filter").addEventListener("change", event => {
  const filter = event.target.value;
  visibleIndices = filter === "all" ? allIndices : allIndices.filter(i => correct(i) === (filter === "correct"));
  page = 0;
  renderGallery();
});
for (const [id, increment] of [["previous", -1], ["next", 1]]) {
  element(id).addEventListener("click", () => {
    page += increment;
    renderGallery();
  });
}
element("digit-training").textContent = mnist.nTrain.toLocaleString("en-US");
element("digit-testing").textContent = mnist.truth.length;
element("digit-accuracy").textContent = percent(allIndices.filter(correct).length / mnist.truth.length);
element("digit-loss").textContent = mnist.logLoss.toFixed(3);
renderGallery();
element("digit-explorer").hidden = false;

function renderThreshold() {
  const threshold = Number(element("cutoff").value);
  const row = wisconsin.thresholds.indexOf(threshold);
  const [tp, fn, fp, tn] = wisconsin.counts[row];
  element("cutoff-value").textContent = threshold.toFixed(2);
  for (const [id, count] of Object.entries({tp, fn, fp, tn})) element(id).textContent = count;
  element("binary-metrics").textContent = `Accuracy ${percent((tp + tn) / wisconsin.nTest)} · Sensitivity ${percent(tp / (tp + fn))} · Specificity ${percent(tn / (tn + fp))}`;
}
element("wdbc-training").textContent = wisconsin.nTrain;
element("wdbc-testing").textContent = wisconsin.nTest;
element("wdbc-loss").textContent = wisconsin.logLoss.toFixed(3);
element("wdbc-brier").textContent = wisconsin.brier.toFixed(3);
element("cutoff").addEventListener("input", renderThreshold);
renderThreshold();
element("binary-explorer").hidden = false;
