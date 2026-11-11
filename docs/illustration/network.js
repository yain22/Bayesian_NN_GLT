/* The displayed graph uses the paper's example with three inputs and two ReLU layers. */
(() => {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const instances = new WeakMap();
  let nextId = 0;

  const color = {
    ink: "#253a40", muted: "#5c6f73", line: "#b8c8c9", outline: "#6d8184",
    blue: "#146b8c", blueFill: "#dff1f6", rust: "#a44822", rustFill: "#f8e9df",
    white: "#ffffff", ghost: "#81908f"
  };

  function el(tag, attrs = {}, text) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  const styles = `
    .net-heading,.net-subheading,.net-unit-label,.net-input-label,.net-bias-label,
    .net-output-label{font-family:Inter,"Segoe UI",Arial,sans-serif;text-anchor:middle}
    .net-heading{fill:${color.ink};font-weight:600}
    .net-subheading{fill:${color.muted};font-weight:400}
    .net-unit-label,.net-input-label,.net-bias-label,.net-output-label{
      dominant-baseline:central;pointer-events:none}
    .net-unit{cursor:pointer;outline:none}
    .net-unit .net-ring{fill:none;stroke:transparent;stroke-width:2}
    .net-unit:hover .net-ring{stroke:${color.outline};stroke-opacity:.55}
    .net-unit:focus-visible .net-ring{stroke:${color.ink};stroke-opacity:1;stroke-width:2.5}
    .net-hit{fill:transparent;stroke:none}
    .net-blue-arrow,.net-orange-arrow{pointer-events:none}
  `;

  function axes(width) {
    const small = width < 505;
    const start = small ? 72 : 67;
    const step = small ? 43 : 37;
    const end = start + step * 7;
    const mid = (start + end) / 2;
    const xs = small
      ? [width * .108, width * .365, width * .642, width * .902]
      : [width * .096, width * .362, width * .655, width * .906];
    return {
      small, xs, start, end, biasY: end + 38, height: end + 61,
      ys: [
        [mid - step * 1.85, mid, mid + step * 1.85],
        Array.from({length: 4}, (_, n) => mid + (n - 1.5) * step * 1.65),
        Array.from({length: 8}, (_, n) => start + n * step), [mid]
      ]
    };
  }

  function drawBell(parent, start, end, fraction, sd, stroke) {
    // Compress the range of widths so an unusually large scale still fits on its arrow.
    const halfWidth = 5 + 19 * sd / (1 + sd);
    const height = 117 / halfWidth;
    const angle = Math.atan2(end.y - start.y, end.x - start.x) * 180 / Math.PI;
    const x = start.x + (end.x - start.x) * fraction;
    const y = start.y + (end.y - start.y) * fraction;
    const path = [`M${-halfWidth} 0`];
    for (let i = -12; i <= 12; i++) {
      const t = i / 12;
      path.push(`L${(halfWidth*t).toFixed(2)} ${(-height*Math.exp(-4.5*t*t)).toFixed(2)}`);
    }
    path.push(`L${halfWidth} 0Z`);
    parent.append(el("path", {class:"net-weight-bell", d:path.join(" "),
      transform:`translate(${x} ${y}) rotate(${angle})`,
      fill:color.white,stroke,"stroke-width":1.5,"stroke-linejoin":"round"}));
  }

  function drawEdge(parent, from, to, highlighted, layer, sd, sourceIndex) {
    const dx = to.x - from.x, dy = to.y - from.y;
    const length = Math.hypot(dx, dy);
    const ux = dx / length, uy = dy / length;
    const startRadius = from.square ? from.r / Math.max(Math.abs(ux), Math.abs(uy)) : from.r;
    const x1 = from.x + startRadius * ux, y1 = from.y + startRadius * uy;
    const x2 = to.x - to.r * ux, y2 = to.y - to.r * uy;
    const stroke = highlighted ? (layer === 1 ? color.blue : color.rust) : color.line;
    const line = el("line", {x1, y1, x2, y2, stroke,
      "stroke-width": highlighted ? 2.5 : 1.05,
      "stroke-opacity": highlighted ? 1 : .75,
      "class": highlighted ? (layer === 1 ? "net-blue-arrow" : "net-orange-arrow") : "net-edge"});
    if (highlighted && layer === 2) line.setAttribute("stroke-dasharray", "5 3");
    parent.append(line);
    if (!highlighted) return;
    if (Number.isFinite(sd) && sd > 0) {
      const fraction = [0.33,0.41,0.35,0.43,0.36][sourceIndex];
      drawBell(parent,{x:x1,y:y1},{x:x2,y:y2},fraction,sd,stroke);
    }
    const tipX = x1 + (x2 - x1) * .62, tipY = y1 + (y2 - y1) * .62;
    const backX = tipX - 7 * ux, backY = tipY - 7 * uy;
    parent.append(el("polygon", {points:
      `${tipX},${tipY} ${backX - 3*uy},${backY + 3*ux} ${backX + 3*uy},${backY - 3*ux}`,
      fill:stroke}));
  }

  function unitButton(point, layer, index, selected, removed, onSelect) {
    const {x, y, r} = point;
    const accent = layer === 1 ? color.blue : color.rust;
    const fill = layer === 1 ? color.blueFill : color.rustFill;
    const name = `Layer ${layer}, neuron ${index}${removed ? ", pruned in this example" : ""}`;
    const g = el("g", {
      class:"net-unit", tabindex:0, role:"button", "aria-label":name,
      "aria-pressed":selected, "data-net-id":`${layer}-${index}`,
      transform:`translate(${x} ${y})`
    });
    g.append(el("title", {}, name), el("rect", {
      x:-22, y:-21, width:44, height:42, rx:8, class:"net-hit"
    }), el("circle", {r:r+4, class:"net-ring"}));
    const shape = el("circle", {r, fill:removed ? color.white : selected ? fill : color.white,
      stroke:removed ? color.ghost : selected ? accent : color.outline,
      "stroke-width":selected ? 2.2 : 1.4});
    if (removed) shape.setAttribute("stroke-dasharray", "3 3");
    g.append(shape);
    g.append(el("text", {x:0, y:0, class:"net-unit-label",
      "font-size":selected ? 13 : 11.5, "font-weight":selected ? 700 : 500,
      fill:removed ? color.muted : selected ? accent : color.ink}, removed ? "×" : index));
    g.addEventListener("click", () => onSelect(layer,index));
    g.addEventListener("keydown", event => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      if (!event.repeat) onSelect(layer,index);
    });
    return g;
  }

  function render(svg, instance, actualWidth) {
    const {state, onSelect} = instance;
    const width = Math.round(Math.min(850, Math.max(300, actualWidth || 720)));
    const geom = axes(width);
    const {xs, ys, small} = geom;
    instance.lastWidth = Math.round(actualWidth);
    const focusId = svg.contains(document.activeElement)
      ? document.activeElement.getAttribute("data-net-id") : null;
    svg.setAttribute("viewBox", `0 0 ${width} ${geom.height}`);
    svg.setAttribute("role", "group");
    svg.setAttribute("aria-labelledby", `net-title-${instance.id}`);
    svg.setAttribute("aria-describedby", `net-desc-${instance.id}`);
    svg.style.display = "block";
    svg.style.width = "100%";
    svg.style.height = "auto";
    const pruned = Boolean(state.pruned);
    const root = el("g");
    const guide = el("g", {"aria-hidden":"true"});
    const inactive = el("g", {"aria-hidden":"true"});
    const active = el("g", {"aria-hidden":"true"});
    const nodes = el("g");
    const headers = ["Inputs", "ReLU layer 1", "ReLU layer 2", "Output"];
    const countLabel = layer => {
      const n = ys[layer].length;
      return pruned && state.layer === layer ? `${n-1} retained` : (layer === 1 ? "K₁ = " : "K₂ = ") + n;
    };
    const subtitles = ["p = 3", countLabel(1), countLabel(2), "Affine"];
    for (let i = 0; i < headers.length; i++) {
      guide.append(el("text", {x:xs[i],y:20,class:"net-heading","font-size":small ? 12 : 15},headers[i]));
      guide.append(el("text", {x:xs[i],y:38,class:"net-subheading","font-size":small ? 10.5 : 12},subtitles[i]));
    }
    const points = ys.map((heights,col) => heights.map((y,j) => ({
      x:xs[col], y, r:col===3 ? (small ? 16 : 20) : (col===0 ? 12 : 13.5),
      removed:pruned && col===state.layer && j+1===state.unit
    })));
    const biases = xs.slice(0,3).map(x=>({x,y:geom.biasY,r:10,square:true}));
    for (let target = 1; target <= 3; target++) {
      const sources = [...points[target-1], biases[target-1]];
      points[target].forEach((p,j) => {
        if (p.removed) return;
        const highlight = target===state.layer && j+1===state.unit;
        sources.forEach((source,i) => {
          if (!source.removed) drawEdge(highlight ? active : inactive, source,p,
            highlight,target,small ? undefined : state.priorSd,i);
        });
      });
    }
    const subscripts = ["₁","₂","₃"];
    points[0].forEach(({x,y,r},i) => {
      nodes.append(el("circle",{cx:x,cy:y,r,fill:color.white,stroke:color.outline,"stroke-width":1.3}),
        el("text",{x,y,class:"net-input-label",fill:color.ink,"font-size":11.5},`x${subscripts[i]}`));
    });
    for (let layer = 1; layer <= 2; layer++) {
      points[layer].forEach((p,j) => nodes.append(unitButton(p,layer,j+1,
        state.layer===layer && state.unit===j+1, p.removed,onSelect)));
    }
    const out = points[3][0];
    const readout = el("text",{x:out.x,y:out.y,class:"net-output-label",fill:color.ink,"font-size":17,"font-style":"italic"},"f");
    readout.append(el("tspan",{"baseline-shift":"sub","font-size":10},"𝒲"));
    nodes.append(el("circle",{cx:out.x,cy:out.y,r:out.r,fill:color.white,stroke:color.ink,"stroke-width":1.4}),readout);
    biases.forEach(({x,y,r}) => {
      const bias = el("g");
      bias.append(el("title",{},"Constant input 1; its arrow carries a bias coefficient."),
        el("rect",{x:x-r,y:y-r,width:r*2,height:r*2,fill:color.white,stroke:color.outline,"stroke-width":1.2}),
        el("text",{x,y,class:"net-bias-label",fill:color.ink,"font-size":12},"1"));
      nodes.append(bias);
    });
    root.append(guide,inactive,active,nodes);
    const description = pruned
      ? `Layer ${state.layer}, neuron ${state.unit} is pruned. Its dashed position has no connections.`
      : `Incoming weights and bias are highlighted for layer ${state.layer}, neuron ${state.unit}.`;
    svg.replaceChildren(el("style",{},styles),
      el("title",{id:`net-title-${instance.id}`},"Interactive two-layer regression network"),
      el("desc",{id:`net-desc-${instance.id}`}, description+" Choose a hidden neuron using a pointer or Tab and Enter."),
      root);
    if (focusId) svg.querySelector(`[data-net-id="${focusId}"]`)?.focus({preventScroll:true});
  }

  window.drawNetwork = function drawNetwork(svg, state, onSelect) {
    let instance = instances.get(svg);
    if (!instance) {
      instance = {id:++nextId};
      instances.set(svg,instance);
      if (window.ResizeObserver) {
        instance.observer = new ResizeObserver(entries => {
          const width = Math.round(entries[0].contentRect.width);
          if (width && width !== instance.lastWidth) render(svg,instance,width);
        });
        instance.observer.observe(svg);
      }
    }
    instance.state = state;
    instance.onSelect = onSelect;
    render(svg,instance,Math.round(svg.getBoundingClientRect().width));
  };
})();
