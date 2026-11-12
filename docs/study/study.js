"use strict";

const byId = id => document.getElementById(id);
const methodNames = {g:"G", hs:"GL", glt:"GLT", fnn:"FNN", dropout:"Dropout", l1:"L1"};
const configs = [
  [300,100,10,.3], [300,100,10,.3], [300,100,10,.3], [150,100,10,.3],
  [600,100,10,.3], [300,500,10,.3], [300,100,30,.3], [300,100,10,.7]
];
function format(value) {
  return value !== 0 && Math.abs(value) < .0005 ? value.toExponential(1) : value.toFixed(3);
}
function item(tag, text, styleClass) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (styleClass) element.className = styleClass;
  return element;
}
function drawScales(data) {
  const quantity = byId("quantity").value;
  const between = quantity === "active_minus_inactive";
  byId("interpret").textContent = between
    ? "In each fit, subtract the median log δ for connections from inactive predictors from the median for connections from active predictors. A positive difference means active inputs typically receive larger scales."
    : "In each first-layer neuron, subtract the 10th percentile of log δ from the 90th percentile. Take the median across neurons. A larger range indicates more variation in shrinkage among connections entering the same neuron.";
  const models = ["g","hs","glt"];
  const values = models.map(key => data.scales[key][quantity]);
  const low = Math.min(0, ...values.map(x => x[0] - 2*x[1]));
  const high = Math.max(0, ...values.map(x => x[0] + 2*x[1]));
  const padding = .08 * (high - low || 1);
  const a = low - padding, b = high + padding;
  const percent = x => 100 * (x-a) / (b-a);
  byId("plot-title").textContent = between ? "Difference in median log δ" : "Median within-neuron interdecile range of log δ";
  byId("lo").textContent = format(a);
  byId("hi").textContent = format(b);
  const rows = models.map((model,i) => {
    const [value,se] = values[i];
    const row = item("div",undefined,"row");
    const track = item("span",undefined,"track");
    const zero = item("i",undefined,"zero");
    zero.style.left = percent(0)+"%";
    const whisker = item("i",undefined,"whisker");
    whisker.style.left = percent(value-2*se)+"%";
    whisker.style.width = (percent(value+2*se)-percent(value-2*se))+"%";
    const point = item("i",undefined,"point");
    point.style.left = percent(value)+"%";
    track.append(zero,whisker,point);
    row.append(item("span",methodNames[model]),track,item("span",format(value)));
    row.title = methodNames[model] + " · Mean " + format(value) + ", MCSE " + format(se);
    return row;
  });
  byId("series").replaceChildren(...rows);
  byId("plot-description").textContent = models.map((m,i) =>
    methodNames[m]+", estimate "+format(values[i][0])+", Monte Carlo standard error "+format(values[i][1])).join(". ");
}
function drawResults() {
  const [cell,snr,depth,width] = ["cell","snr","depth","width"].map(k=>Number(byId(k).value));
  const config = configs[cell-1];
  byId("design").textContent = "C"+cell+" · "+config[0]+" training observations · "+config[2]+" active predictors among "+config[1]+" · predictor AR(1) correlation "+config[3]+".";
  const data = studyMeasurements[[cell,snr,depth,width].join("/")];
  drawScales(data);
  const target = byId("target").value;
  const rows = Object.entries(methodNames).map(([method,label]) => {
    const tr = item("tr");
    const th = item("th",label);
    th.scope = "row";
    const [loss,difference,mcse] = data.loss[method][target];
    tr.append(th,item("td",format(loss)),item("td",difference===null?"Reference":format(difference)),
              item("td",mcse===null?"—":format(mcse)));
    return tr;
  });
  byId("errors").replaceChildren(...rows);
}
if (typeof studyMeasurements === "object") {
  document.querySelectorAll(".controls select, #quantity, #target").forEach(control => control.addEventListener("change",drawResults));
  drawResults();
} else {
  byId("design").textContent = "The study measurements could not be loaded. Reload the page or open the C2 replication script below.";
  document.querySelector(".controls").hidden = true;
  byId("shrinkage").closest("section").hidden = true;
  byId("prediction").closest("section").hidden = true;
  byId("download-tables").hidden = true;
}
