/* Classification readouts for a fixed input and the eight activations of layer 2. */
(() => {
  "use strict";

  const svgNS = "http://www.w3.org/2000/svg";
  const instances = new WeakMap();
  const ink = "#253a40", muted = "#5b6d71", faint = "#c8d4d3";
  const purple = "#6e467d", lavender = "#f3edf5", teal = "#155d75";
  const font = 'Inter,"Segoe UI",Arial,sans-serif';
  const serif = 'Georgia,"Times New Roman",serif';

  function el(name, attrs, content) {
    const node = document.createElementNS(svgNS, name);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
    if (content !== undefined) node.textContent = content;
    return node;
  }

  function label(x, y, content, size = 12, color = ink) {
    return el("text", {x, y, "text-anchor":"middle", "dominant-baseline":"central",
      "font-family":font, "font-size":size, fill:color, "pointer-events":"none"}, content);
  }

  function mathLabel(x, y, letter, sub, suffix = "") {
    const text = label(x, y, letter, 16);
    text.setAttribute("font-family", serif);
    text.setAttribute("font-style", "italic");
    if (sub) text.append(el("tspan", {"baseline-shift":"sub", "font-size":10}, sub));
    if (suffix) text.append(el("tspan", {"font-size":12, "font-style":"normal"}, suffix));
    return text;
  }

  function line(x1, y1, x2, y2, stroke, width) {
    return el("line", {x1, y1, x2, y2, stroke, "stroke-width":width, "stroke-linecap":"round"});
  }

  function button(node, attribute, index, selected, description, handler) {
    node.setAttribute(attribute, index);
    node.setAttribute("tabindex", "0");
    node.setAttribute("role", "button");
    node.setAttribute("aria-label", description);
    node.setAttribute("aria-pressed", String(selected));
    node.setAttribute("class", "classnet-button");
    node.append(el("title", {}, description));
    node.addEventListener("click", () => handler(index));
    node.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        if (!event.repeat) handler(index);
      }
    });
  }

  function render(svg, instance, actualWidth) {
    const {state, onClass, onCoordinate} = instance;
    const categorical = state.family === "categorical";
    const selectedClass = categorical ? state.classIndex : 2;
    const free = !categorical || selectedClass < 10;
    const width = Math.max(260, Math.min(830, Math.round(actualWidth || 640)));
    const compact = width < 450;
    const gap = compact ? 37 : 35;
    const top = 51, bottom = top + 9 * gap, mid = (top + bottom)/2;
    const xSource = .105 * width, xLogit = .38 * width;
    const xLink = .63 * width, xProb = .875 * width;
    const sourceY = j => top + gap * (j - .5);
    const classY = c => top + gap * (c - 1);
    const scoreY = categorical ? classY(selectedClass) : mid;
    const sourceRadius = 14, logitRadius = categorical ? 16 : 21;
    const linkHalf = compact ? 23 : 33, probHalf = Math.min(width*.114, 49);
    const focus = svg.contains(document.activeElement) ? document.activeElement.dataset.focus : null;

    instance.width = Math.round(actualWidth);
    svg.setAttribute("viewBox", `0 0 ${width} ${bottom+35}`);
    svg.setAttribute("role", "group");
    svg.setAttribute("aria-label", categorical ? "Ten-class network readout with softmax probabilities" :
      "Binary network readout with a logistic event probability");
    svg.style.display = "block";
    svg.style.width = "100%";
    svg.style.height = "auto";
    const guide = el("g", {"aria-hidden":"true"}), links = el("g", {"aria-hidden":"true"});
    const weights = el("g", {"aria-hidden":"true"}), sources = el("g"), heads = el("g"), outputs = el("g");
    for (const [x, name] of [[xSource,"Layer 2"],[xLogit,categorical ? "Logits" : "Log odds"],
      [xLink,"Link"],[xProb,compact ? "Probability" : "Class probability"]]) {
      const text = label(x, 16, name, compact ? 11 : 14);
      text.setAttribute("font-weight", 600);
      guide.append(text);
    }

    // Nine free heads read all nine coordinates. Keep the other heads' edges
    // faint so the selected head's coefficients can be traced individually.
    if (categorical) {
      const background = el("g", {opacity:.24, "aria-hidden":"true"});
      for (let c=1; c<10; c++) {
        if (c === selectedClass) continue;
        for (let j=1; j<=9; j++) background.append(
          line(xSource+sourceRadius,sourceY(j),xLogit-logitRadius,classY(c),faint,.85));
      }
      links.append(background);
    }
    for (let j=1; j<=9; j++) {
      const y = sourceY(j), chosen = free && j === state.coordinate;
      const source = el("g", {transform:`translate(${xSource},${y})`, "data-focus":`source-${j}`});
      const activation = state.activations?.[j-1];
      if (free) {
        const start = xSource + sourceRadius, end = xLogit - logitRadius;
        weights.append(line(start, y, end, scoreY, chosen ? purple : "#cab8ce", chosen ? 3.5 : 1.3));
        const hit = line(start, y, end, scoreY, "transparent", 12);
        hit.setAttribute("data-classification-edge", j);
        hit.setAttribute("pointer-events", "stroke");
        hit.style.cursor = "pointer";
        hit.addEventListener("click", () => onCoordinate(j));
        weights.append(hit);
        const sourceName = j === 9 ? "Select the output bias at constant input 1" :
          `Select the output weight from layer 2 unit ${j}` +
          (Number.isFinite(activation) ? `, activation ${activation}` : "");
        button(source, "data-classification-coordinate", j, chosen, sourceName, onCoordinate);
      }
      const paint = {fill:chosen ? lavender : "#fff",stroke:chosen ? purple : muted,
        "stroke-width":chosen ? 2.2 : 1.2,class:"classnet-shape"};
      source.append(el("rect",{x:-18,y:-17,width:36,height:34,fill:"transparent"}));
      source.append(j===9 ? el("rect",{x:-12,y:-12,width:24,height:24,...paint}) :
        el("circle",{r:sourceRadius,...paint}));
      source.append(j===9 ? label(0,0,"1",14) : mathLabel(-1,-1,"h",String(j)+"2"));
      sources.append(source);
    }

    const nHeads = categorical ? 10 : 1;
    for (let c=1; c<=nHeads; c++) {
      const y = categorical ? classY(c) : mid;
      const selected = !categorical || c === selectedClass;
      const fixed = categorical && c === 10;
      links.append(line(xLogit+logitRadius,y,xLink-linkHalf,mid,selected ? purple : faint, selected ? 2.2 : 1));
      const head = el("g", {transform:`translate(${xLogit},${y})`, "data-focus":`logit-${c}`});
      if (categorical) button(head,"data-class-index",c,selected,
        fixed ? "Select class 10, the reference logit fixed at zero" : `Select the class ${c} logit`, onClass);
      const shape = el("circle",{r:logitRadius,fill:selected ? lavender : "#fff",
        stroke:selected ? purple : muted,"stroke-width":selected ? 2.1 : 1.2,class:"classnet-shape"});
      if (fixed) shape.setAttribute("stroke-dasharray", "3 2");
      head.append(shape, fixed ? label(0,0,"0",15) : categorical ?
        mathLabel(-1,-1,"η","i"+c) : mathLabel(0,-1,"f","𝒲"));
      if (fixed) head.append(label(0,logitRadius+8,"Reference",10,muted));
      heads.append(head);
    }

    const probabilities = state.probabilities || [];
    const nClasses = categorical ? 10 : 2;
    for (let c=1; c<=nClasses; c++) {
      const y = categorical ? classY(c) : mid + (c===2 ? -1 : 1)*gap;
      const selected = c === selectedClass;
      links.append(line(xLink+linkHalf,mid,xProb-probHalf,y,selected ? teal : faint, selected ? 2.2 : 1));
      if (!categorical) {
        const branch = label((xLink+xProb)/2-8, y, selected ? "pᵢ" : "1 − pᵢ",12,muted);
        branch.setAttribute("font-family",serif);
        guide.append(branch);
      }
      const entry = el("g", {transform:`translate(${xProb},${y})`, "data-focus":`prob-${c}`});
      if (categorical) button(entry,"data-class-probability",c,selected,`Select the probability of class ${c}`,onClass);
      const prob = probabilities[c-1], value = Number.isFinite(prob) ? (prob*100).toFixed(1)+"%" : "";
      entry.append(el("rect",{x:-probHalf,y:-15,width:probHalf*2,height:30,rx:5,
        fill:selected ? "#e3f0f2" : "#fff",stroke:selected ? teal : faint,
        "stroke-width":selected ? 1.8 : 1.2,class:"classnet-shape"}));
      const name = categorical ? "Class "+c : "Y = "+(c-1);
      entry.append(label(0,value ? -6 : 0,name,11,selected ? teal : ink));
      if (value) entry.append(label(0,6,value,10,selected ? teal : muted));
      outputs.append(entry);
    }
    const link = el("g", {"aria-hidden":"true"});
    link.append(el("rect",{x:xLink-linkHalf,y:mid-14,width:2*linkHalf,height:28,rx:5,
      fill:"#edf3f0",stroke:"#849b95","stroke-width":1.3}));
    link.append(label(xLink,mid,categorical ? "softmax" : "logistic",compact ? 10.5 : 13));
    const desc = categorical ?
      "Each of the first nine logits takes all eight activations and a bias. The tenth logit is fixed at zero. Softmax uses all ten logits for every probability." :
      "The eight hidden activations and bias enter one log odds. The logistic function gives the probability of one. Its complement gives the probability of zero.";
    svg.replaceChildren(el("desc", {}, desc), el("style", {}, `
      .classnet-button{cursor:pointer;outline:none}
      .classnet-button:hover .classnet-shape,.classnet-button:focus-visible .classnet-shape{stroke:#172f39;stroke-width:2.8}
    `), guide, links, weights, sources, heads, link, outputs);
    if (focus) svg.querySelector(`[data-focus="${focus}"]`)?.focus({preventScroll:true});
  }

  window.drawClassificationNetwork = function(svg, state, onClass, onCoordinate) {
    let instance = instances.get(svg);
    if (!instance) {
      instance = {};
      instances.set(svg, instance);
      if (window.ResizeObserver) {
        instance.observer = new ResizeObserver(([entry]) => {
          const width = Math.round(entry.contentRect.width);
          if (width && width !== instance.width) render(svg, instance, width);
        });
        instance.observer.observe(svg);
      }
    }
    Object.assign(instance, {state,onClass,onCoordinate});
    render(svg, instance, svg.getBoundingClientRect().width);
  };
})();
