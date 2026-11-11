/* Conditional prior illustrations. All distances in the tail plot use a fixed layer scale. */
(() => {
  "use strict";

  const ns = "http://www.w3.org/2000/svg";
  const blue = "#135d7a";
  const rust = "#ac4d27";
  const ink = "#273f46";
  const muted = "#53686b";
  const grid = "#d9e1dd";
  const axis = "#8c9d99";
  const ids = new WeakMap();
  let nextId = 1;

  function element(parent, name, attributes, content) {
    const item = document.createElementNS(ns, name);
    for (const [key, value] of Object.entries(attributes)) {
      item.setAttribute(key, String(value));
    }
    if (content !== undefined) item.textContent = content;
    parent.appendChild(item);
    return item;
  }

  function label(svg, x, y, value, attributes = {}) {
    return element(svg, "text", {
      x, y, fill: muted, "font-size": 12, ...attributes
    }, value);
  }

  function line(svg, x1, y1, x2, y2, attributes = {}) {
    return element(svg, "line", {
      x1, y1, x2, y2, stroke: grid, "stroke-width": 1, ...attributes
    });
  }

  function begin(svg, title, description, height) {
    if (!ids.has(svg)) ids.set(svg, "prior-plot-" + nextId++);
    const id = ids.get(svg);
    svg.replaceChildren();
    svg.setAttribute("viewBox", "0 0 440 " + height);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-labelledby", id + "-title");
    svg.setAttribute("aria-describedby", id + "-desc");
    svg.setAttribute("font-family", "inherit");
    svg.style.display = "block";
    svg.style.width = "100%";
    svg.style.height = "auto";
    element(svg, "title", {id: id + "-title"}, title);
    element(svg, "desc", {id: id + "-desc"}, description);
    return id;
  }

  function pathFor(start, end, steps, toX, toY) {
    const points = [];
    for (let i = 0; i <= steps; i++) {
      const v = start + (end - start) * i / steps;
      points.push((i === 0 ? "M" : "L") + toX(v).toFixed(2) + " " + toY(v).toFixed(2));
    }
    return points.join(" ");
  }

  function decimal(value) {
    return Number(value.toFixed(3)).toString().replace("-", "−");
  }

  function bell(svg, sd, lambda, selectedUnit, layer) {
    if (!Number.isFinite(sd) || sd <= 0) throw new RangeError("The prior spread must be positive.");
    const spread = decimal(sd);
    const clipId = begin(svg, "Conditional weight density for unit " + selectedUnit + " in layer " + layer,
      "A zero-centered Gaussian prior with standard deviation " + spread +
      ". The shaded interval from minus " + spread + " to " + spread +
      " contains about 68 percent of conditional prior mass.", 226) + "-clip";
    const left = 42, right = 427, top = 25, bottom = 176;
    // Sigma varies from 0.25 to 2. Keep both axes fixed over that sweep.
    const unit = lambda;
    const limit = 6 * unit;
    const peak = 1.65 / unit;
    const toX = value => left + (value + limit) * (right - left) / (2 * limit);
    const toY = density => bottom - density * (bottom - top) / peak;
    const density = w => Math.exp(-0.5 * (w / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));
    label(svg, left, 14, "Density", {fill: ink});

    for (const tick of [0, 0.5, 1, 1.5].map(v => v/unit)) {
      const y = toY(tick);
      line(svg, left, y, right, y);
      label(svg, left - 7, y + 4, decimal(tick), {"text-anchor": "end"});
    }
    const defs = element(svg, "defs", {});
    const clip = element(defs, "clipPath", {id: clipId});
    element(clip, "rect", {x: left, y: top, width: right - left, height: bottom - top});
    const curves = element(svg, "g", {"clip-path": "url(#" + clipId + ")"});
    const shaded = pathFor(-sd, sd, 140, toX, w => toY(density(w)));
    element(curves, "path", {
      d: shaded + " L " + toX(sd) + " " + bottom + " L " + toX(-sd) + " " + bottom + " Z",
      fill: "#b8dce6"
    });
    for (const end of [-sd, sd]) {
      line(curves, toX(end), bottom, toX(end), toY(density(end)), {
        stroke: blue, "stroke-width": 1, "stroke-dasharray": "3 3"
      });
    }
    element(curves, "path", {
      d: pathFor(-limit, limit, 420, toX, w => toY(density(w))),
      fill: "none", stroke: blue, "stroke-width": 2.7
    });
    line(svg, left, bottom, right, bottom, {stroke: axis});
    for (const tick of [-4, -2, 0, 2, 4].map(v => v*unit)) {
      line(svg, toX(tick), bottom, toX(tick), bottom + 4, {stroke: axis});
      label(svg, toX(tick), bottom + 18, decimal(tick), {"text-anchor": "middle"});
    }
    const axisTitle = label(svg, (left + right) / 2, 217, "Incoming coefficient ", {"text-anchor": "middle", fill: ink});
    element(axisTitle, "tspan", {"font-family":"Georgia, serif", "font-style":"italic"}, "w");
    element(axisTitle, "tspan", {"baseline-shift":"sub", "font-size":9}, String(selectedUnit) + layer + ",j");
  }

  function tailProbs(xi, threshold = 10) {
    if (!Number.isFinite(xi) || xi <= 0) throw new RangeError("The tail shape must be positive.");
    if (typeof threshold !== "number" || Number.isNaN(threshold)) throw new RangeError("Invalid threshold.");
    if (threshold <= 0) return {g: 1, gl: 1, glt: 1};
    // For t = threshold / layer scale: GL follows a standard half-Cauchy, GLT a unit-scale GPD.
    return {
      g: threshold < 1 ? 1 : 0,
      gl: 2 * Math.atan2(1, threshold) / Math.PI,
      glt: Math.exp(-Math.log1p(xi * threshold) / xi)
    };
  }

  function tail(svg, xi, layer, unit) {
    const probability = tailProbs(xi);
    const percent = p => (100 * p).toFixed(1) + "%";
    const clipId = begin(svg, "Tail probabilities for unit "+unit+" in layer "+layer,
      "Chance that the selected unit scale exceeds a multiple of its layer's scale. The probability axis is logarithmic. " +
      "At ten times the layer scale, global shrinkage gives zero, the grouped horseshoe gives " +
      percent(probability.gl) + ", and GLT with tail shape " + decimal(xi) +
      " gives " + percent(probability.glt) + ".", 246) + "-clip";
    const left = 51, right = 425, top = 44, bottom = 193;
    const toX = t => left + t * (right - left) / 12;
    const toY = p => top + (bottom - top) * Math.log(p) / Math.log(0.02);

    label(svg, left, 15, "Exceedance probability (log scale)", {fill: ink});
    for (const tick of [1, 0.5, 0.2, 0.1, 0.05, 0.02]) {
      line(svg, left, toY(tick), right, toY(tick));
      label(svg, left - 7, toY(tick) + 4, Math.round(tick * 100) + "%", {"text-anchor": "end"});
    }
    line(svg, left, 31, left + 22, 31, {stroke: blue, "stroke-width": 2.6});
    label(svg, left + 28, 35, "GL", {fill: blue, "font-weight": "600"});
    line(svg, left + 79, 31, left + 101, 31, {stroke: rust, "stroke-width": 2.6, "stroke-dasharray": "6 4"});
    label(svg, left + 107, 35, "GLT", {fill: rust, "font-weight": "600"});
    line(svg, left + 164, 31, left + 186, 31, {stroke: ink, "stroke-width": 2.4, "stroke-dasharray":"2 2"});
    label(svg, left + 192, 35, "G", {fill:ink, "font-weight":"600"});
    line(svg, toX(10), top, toX(10), bottom, {stroke: axis, "stroke-dasharray": "2 4"});

    const clip = element(element(svg, "defs", {}), "clipPath", {id: clipId});
    element(clip, "rect", {x: left - 1.5, y: top - 2, width: right - left + 3, height: bottom - top + 3});
    const marks = element(svg, "g", {"clip-path": "url(#" + clipId + ")"});
    line(marks,toX(0),toY(1),toX(1),toY(1),{stroke:ink,"stroke-width":2.4,"stroke-dasharray":"2 2"});
    element(marks,"circle",{cx:toX(1),cy:toY(1),r:2.4,fill:"white",stroke:ink});
    for (const [model, color] of [["gl", blue], ["glt", rust]]) {
      const style = model === "glt" ? {"stroke-dasharray": "6 4"} : {};
      element(marks, "path", {
        d: pathFor(0, 12, 260, toX, t => toY(tailProbs(xi, t)[model])),
        fill: "none", stroke: color, "stroke-width": 2.6, ...style
      });
      element(marks, "circle", {cx: toX(10), cy: toY(probability[model]), r: 3.6, fill: "white", stroke: color, "stroke-width": 2});
    }
    line(svg, left, bottom, right, bottom, {stroke: axis});
    for (const tick of [0, 2, 4, 6, 8, 10, 12]) {
      line(svg, toX(tick), bottom, toX(tick), bottom + 4, {stroke: axis});
      label(svg, toX(tick), bottom + 18, String(tick), {"text-anchor": "middle"});
    }
    label(svg, (left + right) / 2, 236, "Threshold multiple z", {"text-anchor": "middle", fill: ink});
  }

  window.PriorPlots = Object.freeze({bell, tail, tailProbs});
})();
