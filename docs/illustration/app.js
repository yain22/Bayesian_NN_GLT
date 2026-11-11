/* All panels share one selected unit and one set of conditional prior parameters. */
(() => {
  "use strict";
  const item = id => document.getElementById(id);
  const all = selector => [...document.querySelectorAll(selector)];
  const tabs = all(".tab");
  const layers = [
    null,
    {tau: .5, xi: 1, u: [.25, .60, .40, .73]},
    {tau: .5, xi: 1, u: [.35, .58, .24, .72, .60, .48, .31, .68]}
  ];
  const state = {layer:1, unit:2, sigma:1, prior:"GL", form:"original", method:"ghosh", chance:95, topic:"spread"};
  const fmt = v => v >= .01 ? v.toFixed(2) : v.toPrecision(2);
  const sub = n => String(n).replace(/[0-9]/g, ch => "₀₁₂₃₄₅₆₇₈₉"[Number(ch)]);
  const indexed = (letter,k,l) => letter + sub(k) + sub(l);

  function unitScale(layer, u) {
    if (state.prior === "G") return layer.tau;
    if (state.prior === "GL") return layer.tau * Math.tan(Math.PI * u / 2);
    return layer.tau * Math.expm1(-layer.xi * Math.log1p(-u)) / layer.xi;
  }

  function selectUnit(l,k) {
    state.layer = l;
    state.unit = k;
    render();
  }

  function network(layer, sd) {
    const l = state.layer, k = state.unit, d = l === 1 ? 4 : 5;
    const pruned = state.topic === "prune" && state.method === "ghosh" && state.chance > 90;
    const externalRule = state.topic === "prune" && state.method === "ghosh";
    window.drawNetwork(item("network"), {layer:l, unit:k, pruned, priorSd:externalRule ? undefined : sd},selectUnit);
    item("selection-name").textContent = "Layer l = " + l + " · Unit k = " + k;
    const vector = item("selected-vector");
    const term = (letter,index,bold=false) => {
      const word = document.createElement(bold ? "b" : "i");
      word.textContent = letter;
      const suffix = document.createElement("sub");
      suffix.textContent = index;
      word.append(suffix);
      return word;
    };
    const pair = String(k) + l;
    const weights = [];
    for (let j=1; j<d; j++) weights.push(term("w", pair+","+j), ", ");
    const transpose = document.createElement("sup");
    transpose.textContent = "T";
    vector.replaceChildren(term("w",pair,true), " = (", ...weights, term("b",pair), ")", transpose);
    item("selection-detail").textContent = externalRule
      ? "Grouped-horseshoe pruning example. " + (pruned ? "The dashed node marks the removed unit." : "The selected unit remains connected.")
      : "Layer width K" + sub(l) + " = " + layer.u.length + ". Incoming dimension d" + sub(l) + " = " + d +
        ", including the bias. Under " + state.prior + ", σ" + indexed("λ",k,l) + " = " + fmt(sd) + ".";
  }

  function spread(layer, lambda, sd) {
    item("spread").value = state.sigma;
    item("spread-value").value = fmt(state.sigma);
    window.PriorPlots.bell(item("bell-chart"),sd,lambda);
    item("spread-readout").textContent = "Conditional on " + indexed("λ",state.unit,state.layer) + " = " +
      fmt(lambda) + " and σ = " + fmt(state.sigma) + ", one coordinate has SD " + fmt(sd) +
      ". Its central 68.27% interval is [−" + fmt(sd) + ", " + fmt(sd) + "].";
  }

  function selectPriorButtons() {
    all("[data-prior]").forEach(b => b.setAttribute("aria-pressed", b.dataset.prior === state.prior));
  }

  function scales(layer) {
    selectPriorButtons();
    all("[data-law]").forEach(b => b.hidden = b.dataset.law !== state.prior);
    item("layer-scale").value = layer.tau;
    item("layer-scale-value").value = fmt(layer.tau);
    item("percentile-control").hidden = state.prior === "G";
    const percent = Math.round(100 * layer.u[state.unit-1]);
    item("percentile").value = percent;
    item("percentile-value").value = percent+"%";
    const sds = layer.u.map(u => state.sigma * unitScale(layer,u));
    const max = Math.max(1,Math.ceil(Math.max(...sds)));
    const bars = item("unit-bars"), focused = bars.contains(document.activeElement);
    const span = (cls,txt) => {const s=document.createElement("span");s.className=cls;if(txt)s.textContent=txt;return s;};
    bars.replaceChildren(...sds.map((sd,i) => {
      const button = document.createElement("button");
      button.type = "button"; button.className = "scale-row";
      button.setAttribute("aria-pressed",state.unit === i+1);
      button.setAttribute("aria-label","Unit "+(i+1)+", conditional prior standard deviation "+fmt(sd));
      const track=span("scale-track"), fill=span("scale-fill");
      fill.style.width = (100*sd/max)+"%";
      if (state.layer === 2) fill.style.backgroundColor = "var(--rust)";
      track.append(fill);
      button.append(span("scale-name","k = "+(i+1)),track,span("scale-value",fmt(sd)));
      button.addEventListener("click",() => selectUnit(state.layer,i+1));
      return button;
    }));
    if (focused) bars.children[state.unit-1].focus({preventScroll:true});
    item("bars-caption").textContent = "Common axis 0–"+max+" in weight units. σ = "+fmt(state.sigma)+
      (state.prior === "GLT" ? ", ξ"+sub(state.layer)+" = "+fmt(layer.xi)+"." : ".")+
      (state.prior === "G" ? "" : " Quantiles are illustrative.");
  }

  function hierarchy() {
    selectPriorButtons();
    all("[data-form]").forEach(b => b.setAttribute("aria-pressed",b.dataset.form === state.form));
    all("[data-hierarchy]").forEach(b => b.hidden = b.dataset.hierarchy !== state.prior+":"+state.form);
    item("hierarchy-title").textContent = state.form === "expanded" ? "Equivalent hierarchy with parameter expansion" : "Original hierarchy";
    window.drawHierarchy(item("hierarchy-graph"),state.prior,state.form);
  }

  function tail(layer) {
    item("tail-shape").value = layer.xi;
    item("tail-value").value = fmt(layer.xi);
    window.PriorPlots.tail(item("tail-chart"),layer.xi);
    const probabilities = window.PriorPlots.tailProbs(layer.xi);
    for (const name of ["g","gl","glt"]) item("tail-"+name).textContent = (100*probabilities[name]).toFixed(name==="g"?0:1)+"%";
  }

  function pruning(layer) {
    all("[data-method]").forEach(b => b.setAttribute("aria-pressed",b.dataset.method === state.method));
    const fixed = state.method === "ours";
    item("prune-chance").disabled = fixed;
    item("prune-value").value = fixed ? "—" : state.chance+"%";
    const id = "Unit "+state.unit+" in layer "+state.layer;
    item("prune-result").textContent = state.method === "ours" ? id+" stays. Each fitted architecture retains all its units."
      : state.chance > 90 ? id+" is pruned. This leaves "+(layer.u.length-1)+" units in that layer."
      : id+" stays. The fitted probability must strictly exceed 90% for pruning.";
  }

  function render() {
    const layer = layers[state.layer], lambda = unitScale(layer,layer.u[state.unit-1]), sd=state.sigma*lambda;
    network(layer,sd);
    if (state.topic === "spread") spread(layer,lambda,sd);
    if (state.topic === "scales") scales(layer);
    if (state.topic === "hierarchy") hierarchy();
    if (state.topic === "tail") tail(layer);
    if (state.topic === "prune") pruning(layer);
  }

  function showTopic(topic) {
    state.topic = topic;
    for (const tab of tabs) {
      const active = tab.id === "tab-"+topic;
      tab.setAttribute("aria-selected",active); tab.tabIndex = active ? 0 : -1;
      item(tab.getAttribute("aria-controls")).hidden = !active;
    }
    render();
  }
  tabs.forEach((tab,i) => {
    tab.addEventListener("click",() => showTopic(tab.id.slice(4)));
    tab.addEventListener("keydown",event => {
      const next = {ArrowRight:(i+1)%tabs.length,ArrowLeft:(i+tabs.length-1)%tabs.length,Home:0,End:tabs.length-1}[event.key];
      if (next === undefined) return;
      event.preventDefault(); tabs[next].focus(); showTopic(tabs[next].id.slice(4));
    });
  });
  for (const attr of ["prior","form","method"]) all("[data-"+attr+"]").forEach(b => b.addEventListener("click",() => {
    state[attr]=b.dataset[attr]; render();
  }));
  const bind = (id,handler) => item(id).addEventListener("input",e=>{handler(e.target.valueAsNumber);render();});
  bind("spread",v=>state.sigma=v);
  bind("layer-scale",v=>layers[state.layer].tau=v);
  bind("percentile",v=>layers[state.layer].u[state.unit-1]=v/100);
  bind("tail-shape",v=>layers[state.layer].xi=v);
  bind("prune-chance",v=>state.chance=v);
  const fragment = location.hash.slice(1);
  showTopic(tabs.some(t=>t.id==="tab-"+fragment) ? fragment : "spread");
})();
