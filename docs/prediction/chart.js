(() => {
  "use strict";
  const data = regressionExample;
  const svg = document.querySelector("#regression-chart");
  const slider = document.querySelector("#location");
  const priorNotes = {
    G: "All eight hidden units share a single shrinkage scale.",
    GL: "Each hidden unit has its own half-Cauchy scale, multiplied by the shared layer scale.",
    GLT: "Each hidden unit has a local scale. The layer learns a tail shape that controls how easily a unit can have a large scale."
  };
  const bandNotes = {
    mean: "Each draw gives the mean at this input under one draw of the weights, converted back to the original response units.",
    response: "Each draw adds Gaussian noise to the fitted mean. The training response standard deviation scales that noise back to the original response units."
  };
  const yValues = [...data.y];
  for (const fit of Object.values(data.fits)) {
    yValues.push(...fit.meanLow, ...fit.meanHigh, ...fit.responseLow, ...fit.responseHigh);
  }
  const yMin = Math.floor(Math.min(...yValues));
  const yMax = Math.ceil(Math.max(...yValues));
  const yStep = yMax - yMin > 8 ? 2 : 1;
  const maxUnitSpread = Math.max(...Object.values(data.fits).flatMap(fit => fit.geometric_prior_sd));

  function showUnits(prior) {
    const chart = document.querySelector("#unit-scales");
    if (chart.dataset.prior === prior) return;
    chart.dataset.prior = prior;
    chart.replaceChildren();
    document.querySelector("#unit-heading").textContent = "Shrinkage across the eight hidden units · " + prior;
    const fit = data.fits[prior];
    document.querySelector("#layer-scale").textContent = fit.geometric_layer_scale.toFixed(3);
    document.querySelector("#tail-summary").hidden = fit.geometric_tail_shape === null;
    document.querySelector("#tail-shape").textContent = fit.geometric_tail_shape?.toFixed(3) ?? "";
    fit.geometric_prior_sd.forEach((sd, k) => {
      const item = document.createElement("li");
      const name = document.createElement("span");
      name.className = "unit-name";
      name.textContent = "Unit " + (k + 1);
      const track = document.createElement("span");
      track.className = "unit-track";
      track.setAttribute("aria-hidden", "true");
      const bar = document.createElement("span");
      bar.className = "unit-fill";
      bar.style.height = (100 * sd / maxUnitSpread) + "%";
      track.append(bar);
      const value = document.createElement("span");
      value.className = "unit-value";
      value.textContent = sd.toFixed(3);
      item.append(name, track, value);
      chart.append(item);
    });
  }

  function element(tag, attributes, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    svg.append(node);
    return node;
  }

  function render() {
    const prior = document.querySelector("input[name=prior]:checked").value;
    const band = document.querySelector("input[name=band]:checked").value;
    const fit = data.fits[prior];
    const low = fit[band + "Low"];
    const high = fit[band + "High"];
    const i = Number(slider.value);
    const activeX = data.grid[i];
    const width = Math.max(230, svg.clientWidth);
    const height = Math.round(Math.max(255, Math.min(385, width * .58)));
    const left = 39, right = width - 15, top = 26, bottom = height - 45;
    const toX = x => left + (x + 2) / 4 * (right - left);
    const toY = y => bottom - (y - yMin) / (yMax - yMin) * (bottom - top);
    const anchor = (x, y) => toX(x).toFixed(2) + "," + toY(y).toFixed(2);
    const points = values => values.map((v, k) => anchor(data.grid[k], v));
    const line = (x1, y1, x2, y2, attributes) => element("line", {x1, y1, x2, y2, ...attributes});
    const label = (x, y, text, position = "middle") => element("text", {
      x, y, fill: "#52676a", "font-size": 12, "text-anchor": position
    }, text);
    svg.replaceChildren();
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    for (let y = yMin; y <= yMax; y += yStep) {
      line(left, toY(y), right, toY(y), {stroke: "#d6dcd7"});
      label(left - 9, toY(y) + 4, y, "end");
    }
    for (const x of [-2, -1, 0, 1, 2]) {
      line(toX(x), bottom, toX(x), bottom + 5, {stroke: "#80918e"});
      label(toX(x), bottom + 20, String(x).replace("-", "−"));
    }
    label(left, 15, "Response y (original units)", "start");
    label((left + right) / 2, height - 4, "Input x");
    element("polygon", {
      points: [...points(high), ...points(low).reverse()].join(" "), fill: "#c0dad7",
      stroke: "#4c7776", "stroke-width": 1.3, "stroke-linejoin": "round"
    });
    line(toX(activeX), top, toX(activeX), bottom, {
      stroke: "#758682", "stroke-dasharray": "4 4"
    });
    element("polyline", {
      points: points(fit.mean).join(" "), fill: "none", stroke: "#155f79",
      "stroke-width": 2.7, "stroke-linejoin": "round"
    });
    data.x.forEach((x, k) => element("circle", {
      cx: toX(x), cy: toY(data.y[k]), r: 3.5, fill: "#a94c2b", stroke: "#fffefa", "stroke-width": .85
    }));
    line(toX(activeX), toY(low[i]), toX(activeX), toY(high[i]),
         {stroke: "#155f79", "stroke-width": 2});
    for (const bound of [low[i], high[i]]) {
      line(toX(activeX) - 4, toY(bound), toX(activeX) + 4, toY(bound), {stroke: "#155f79", "stroke-width": 2});
    }
    element("circle", {cx: toX(activeX), cy: toY(fit.mean[i]), r: 4.5, fill: "#fffefa",
                       stroke: "#155f79", "stroke-width": 2});
    const displayX = activeX.toFixed(2);
    slider.setAttribute("aria-valuetext", displayX);
    document.querySelector("#x-value").textContent = displayX;
    document.querySelector("#mean-value").textContent = fit.mean[i].toFixed(2);
    document.querySelector("#band-value").textContent = low[i].toFixed(2) + " to " + high[i].toFixed(2);
    document.querySelector("#prior-explanation").textContent = priorNotes[prior];
    document.querySelector("#band-explanation").textContent = bandNotes[band];
    document.querySelector("#chart-heading").textContent = "Fitted regression · " + prior;
    showUnits(prior);
  }
  document.querySelectorAll("input[type=radio]").forEach(input => input.addEventListener("change", render));
  slider.addEventListener("input", render);
  window.addEventListener("resize", render);
  render();
})();
