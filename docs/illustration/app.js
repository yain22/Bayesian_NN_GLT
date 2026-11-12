/* The network, hierarchy and plots follow the same selected coefficient. */
(() => {
  "use strict";
  const item = id => document.getElementById(id);
  const all = selector => [...document.querySelectorAll(selector)];
  const tabs = all(".tab");
  function makeLayer(width, dimension) {
    const quantiles = Array.from({length:width}, (_,k) =>
      Array.from({length:dimension}, (_,j) => (20 + (17*k + 11*j + 7) % 58) / 100));
    return {tau:.5, xi:1, width, dimension, quantiles};
  }
  const layers = [null, makeLayer(4,4), makeLayer(8,5)];
  const state = {layer:1, unit:2, j:1, sigma:1, prior:"GL", form:"original", method:"ghosh", chance:95, topic:"spread"};
  const selectedNumbers = all("[data-selection]");
  const selectedFormulas = all("math[data-selection-tex]");
  const layerPassages = all("[data-in-layer]");
  let displayedPair;
  const fmt = v => v >= .01 ? v.toFixed(2) : v.toPrecision(2);
  const sub = n => String(n).replace(/[0-9]/g, ch => "₀₁₂₃₄₅₆₇₈₉"[Number(ch)]);
  const indexed = (letter,k,l,j) => letter + sub(k) + sub(l) + (j === undefined ? "" : "," + sub(j));
  const coordinateName = (j, layer) => j === layer.dimension ? "Bias" :
    (state.layer === 1 ? "Input " : "Layer 1 unit ") + j;

  function coefficientScale(layer, u) {
    if (state.prior === "G") return layer.tau;
    if (state.prior === "GL") return layer.tau * Math.tan(Math.PI * u / 2);
    return layer.tau * Math.expm1(-layer.xi * Math.log1p(-u)) / layer.xi;
  }

  function selectUnit(l,k) {
    const previous = layers[state.layer];
    state.j = state.j === previous.dimension ? layers[l].dimension : Math.min(state.j, layers[l].dimension - 1);
    state.layer = l;
    state.unit = k;
    render();
  }

  function selectCoordinate(j) {
    state.j = j;
    render();
  }

  function bindSelection(layer) {
    const l = state.layer, k = state.unit;
    const inputs = layer.dimension - 1;
    const indices = {l, k, kl:String(k)+l, j:state.j, prev:l-1, width:layer.width, inputs, d:layer.dimension, m:layer.width * layer.dimension};
    const selectionKey = indices.kl + "," + state.j;
    if (displayedPair !== selectionKey) {
      const select = item("coordinate");
      select.replaceChildren(...Array.from({length:layer.dimension}, (_,index) => {
        const option = document.createElement("option");
        option.value = index + 1;
        option.textContent = "j = " + (index + 1) + " · " + coordinateName(index + 1,layer);
        return option;
      }));
      select.value = state.j;
      for (const element of selectedNumbers) element.textContent = indices[element.dataset.selection];
      for (const formula of selectedFormulas) {
        const tex = formula.dataset.selectionTex.replace(/\[\[(\w+)\]\]/g, (_, key) => indices[key]);
        formula.setAttribute("alttext",tex);
      }
      for (const passage of layerPassages) passage.hidden = Number(passage.dataset.inLayer) !== l;
      displayedPair = selectionKey;
    }
    return indices;
  }

  function network(layer, sd) {
    const {l,k,d} = bindSelection(layer);
    const pruned = state.topic === "prune" && state.method === "ghosh" && state.chance > 90;
    item("path-pruned").hidden = !pruned;
    const externalRule = state.topic === "prune" && state.method === "ghosh";
    const active = {
      layer:l, unit:k, pruned, coordinate:externalRule ? undefined : state.j,
      priorSds:externalRule ? [] : layer.quantiles[k-1].map(u => state.sigma * coefficientScale(layer,u))
    };
    window.drawNetwork(item("network"), active, selectUnit, externalRule ? null : selectCoordinate);
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
    item("coordinate").disabled = externalRule;
    item("selection-detail").textContent = externalRule
      ? "Grouped-horseshoe pruning example. " + (pruned ? "The dashed node marks the removed unit." : "The selected unit remains connected.")
      : "Layer width K" + sub(l) + " = " + layer.width + ". Incoming dimension d" + sub(l) + " = " + d +
        ", including the bias. Under " + state.prior + ", σ" + indexed("λ",k,l,state.j) + " = " + fmt(sd) + ".";
  }

  function spread(layer, lambda, sd) {
    item("spread").value = state.sigma;
    item("spread-value").value = fmt(state.sigma);
    window.PriorPlots.bell(item("bell-chart"),sd,lambda,state.unit,state.layer,state.j,layer.dimension);
    item("spread-readout").textContent = "Conditional on " + indexed("λ",state.unit,state.layer,state.j) + " = " +
      fmt(lambda) + " and σ = " + fmt(state.sigma) + ", coordinate " + state.j + " has SD " + fmt(sd) +
      ". Its central 68.27% interval is [−" + fmt(sd) + ", " + fmt(sd) + "].";
  }

  function selectPriorButtons() {
    all("[data-prior]").forEach(b => b.setAttribute("aria-pressed", b.dataset.prior === state.prior));
  }

  function scales(layer) {
    all("[data-law]").forEach(b => b.hidden = b.dataset.law !== state.prior);
    item("layer-scale").value = layer.tau;
    item("layer-scale-value").value = fmt(layer.tau);
    item("percentile-control").hidden = state.prior === "G";
    const u = layer.quantiles[state.unit-1][state.j-1];
    const percent = Math.round(100 * u);
    item("percentile").value = percent;
    item("percentile-value").value = percent+"%";
    const lambda = coefficientScale(layer,u);
    const unitLabel = indexed("λ",state.unit,state.layer,state.j);
    const layerLabel = "τ"+sub(state.layer)+" = "+fmt(layer.tau);
    const substitution = item("scale-substitution");
    if (state.prior === "G") {
      substitution.textContent = unitLabel+" = "+layerLabel+".";
    } else if (state.prior === "GL") {
      substitution.textContent = "At u = "+fmt(u)+", "+indexed("r",state.unit,state.layer,state.j)+" ≈ "+
        (lambda/layer.tau).toPrecision(3)+". With "+layerLabel+", "+unitLabel+" ≈ "+lambda.toPrecision(3)+".";
    } else {
      substitution.textContent = "At u = "+fmt(u)+", "+layerLabel+" and ξ"+sub(state.layer)+" = "+
        fmt(layer.xi)+" give "+unitLabel+" ≈ "+lambda.toPrecision(3)+".";
    }
    const sds = layer.quantiles[state.unit-1].map(u => state.sigma * coefficientScale(layer,u));
    const max = Math.max(1,Math.ceil(Math.max(...sds)));
    const bars = item("unit-bars"), focused = bars.contains(document.activeElement);
    const span = (cls,txt) => {const s=document.createElement("span");s.className=cls;if(txt)s.textContent=txt;return s;};
    bars.replaceChildren(...sds.map((sd,i) => {
      const button = document.createElement("button");
      button.type = "button"; button.className = "scale-row";
      button.setAttribute("aria-pressed",state.j === i+1);
      button.setAttribute("aria-label",coordinateName(i+1,layer)+" of unit "+state.unit+" in layer "+state.layer+", conditional prior standard deviation "+fmt(sd));
      const track=span("scale-track"), fill=span("scale-fill");
      fill.style.width = (100*sd/max)+"%";
      if (state.layer === 2) fill.style.backgroundColor = "var(--rust)";
      track.append(fill);
      button.append(span("scale-name","σ"+indexed("λ",state.unit,state.layer,i+1)),track,span("scale-value",fmt(sd)));
      button.addEventListener("click",() => selectCoordinate(i+1));
      return button;
    }));
    if (focused) bars.children[state.j-1].focus({preventScroll:true});
    item("bars-caption").textContent = "Common axis 0–"+max+" in weight units. σ = "+fmt(state.sigma)+
      (state.prior === "GLT" ? ", ξ"+sub(state.layer)+" = "+fmt(layer.xi)+"." : ".")+
      (state.prior === "G" ? "" : " Quantiles are illustrative.");
  }

  function hierarchy() {
    all("[data-form]").forEach(b => b.setAttribute("aria-pressed",b.dataset.form === state.form));
    all("[data-hierarchy]").forEach(b => b.hidden = b.dataset.hierarchy !== state.prior+":"+state.form);
    item("hierarchy-title").textContent = state.form === "expanded" ? "Equivalent hierarchy with parameter expansion" : "Original hierarchy";
    let weightLaw = "local";
    if (state.prior === "G") weightLaw = state.form === "original" ? "global" : "squared-layer";
    if (state.prior === "GL" && state.form === "expanded") weightLaw = "squared-product";
    all("[data-weight-law]").forEach(node => node.hidden = node.dataset.weightLaw !== weightLaw);
    all("[data-source-layer]").forEach(node => node.hidden = Number(node.dataset.sourceLayer) !== state.layer);
    all("[data-coordinate]").forEach(node => {
      const j = node.dataset.coordinate === "bias" ? layers[state.layer].dimension : Number(node.dataset.coordinate);
      node.classList.toggle("chosen-coordinate", j === state.j);
    });
    window.drawHierarchy(item("hierarchy-graph"),state.prior,state.form,state.layer,state.unit,state.j,layers[state.layer].dimension,layers[state.layer].width);
  }

  function tail(layer) {
    item("tail-shape").value = layer.xi;
    item("tail-value").value = fmt(layer.xi);
    window.PriorPlots.tail(item("tail-chart"),layer.xi,state.layer,state.unit,state.j);
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
      : state.chance > 90 ? id+" is pruned. This leaves "+(layer.width-1)+" units in that layer."
      : id+" stays. The fitted probability must strictly exceed 90% for pruning.";
  }

  function render() {
    selectPriorButtons();
    const layer = layers[state.layer], lambda = coefficientScale(layer,layer.quantiles[state.unit-1][state.j-1]), sd=state.sigma*lambda;
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
  item("coordinate").addEventListener("change", e=>selectCoordinate(Number(e.target.value)));
  bind("spread",v=>state.sigma=v);
  bind("layer-scale",v=>layers[state.layer].tau=v);
  bind("percentile",v=>layers[state.layer].quantiles[state.unit-1][state.j-1]=v/100);
  bind("tail-shape",v=>layers[state.layer].xi=v);
  bind("prune-chance",v=>state.chance=v);
  const fragment = location.hash.slice(1);
  showTopic(tabs.some(t=>t.id==="tab-"+fragment) ? fragment : "spread");
})();
