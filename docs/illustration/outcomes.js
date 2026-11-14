/* Conditional class probabilities for one illustrative input to the readout. */
(() => {
  "use strict";
  const get = id => document.getElementById(id);
  const activations = [1, 0.5, 1.25, 0.75, 0.25, 1.5, 0, 2, 1];
  const binaryHead = new Array(9).fill(0);
  const categoricalHeads = Array.from({length:9}, () => new Array(9).fill(0));
  let family = "binary", classIndex = 1;
  const coordinates = {binary:1, categorical:1};
  const coordinateSelect = get("class-coordinate"), classSelect = get("class-select");
  for (let j = 1; j <= 9; j++) {
    const label = j === 9 ? "Output bias · input 1" : "Unit " + j + " · activation " + activations[j-1];
    coordinateSelect.add(new Option(label, j));
  }
  for (let c = 1; c <= 10; c++) {
    classSelect.add(new Option(c === 10 ? "Class 10 · reference" : "Class " + c, c));
  }

  const score = weights => weights.reduce((total, value, index) => total + value * activations[index], 0);
  function softmax(logits) {
    const max = Math.max(...logits);
    const numerators = logits.map(x => Math.exp(x-max));
    const total = numerators.reduce((a,b) => a+b, 0);
    return numerators.map(x => x/total);
  }

  function displayProbabilities(probabilities, selected) {
    const container = get("class-probabilities");
    if (container.children.length !== probabilities.length) {
      container.replaceChildren(...probabilities.map((_, index) => {
        const row = document.createElement("div");
        row.className = "class-prob-row";
        row.dataset.probabilityClass = index+1;
        const name = document.createElement("span");
        name.textContent = family === "binary" ? "Y = " + index : "Class " + (index+1);
        const track = document.createElement("span");
        track.className = "scale-track";
        track.setAttribute("aria-hidden", "true");
        const fill = document.createElement("span");
        fill.className = "scale-fill";
        track.append(fill);
        row.append(name, track, document.createElement("span"));
        return row;
      }));
    }
    probabilities.forEach((p, index) => {
      const row = container.children[index];
      row.dataset.probability = p;
      row.classList.toggle("inspected-class", index+1 === selected);
      row.querySelector(".scale-fill").style.width = 100*p + "%";
      row.lastElementChild.textContent = (100*p).toFixed(2) + "%";
    });
  }

  function selectClass(c) {
    classIndex = c;
    render();
  }
  function selectCoordinate(j) {
    coordinates[family] = j;
    render();
  }

  function render() {
    const categorical = family === "categorical", j = coordinates[family];
    const selectedClass = categorical ? classIndex : 2;
    const reference = categorical && classIndex === 10;
    const logits = categorical ? [...categoricalHeads.map(score), 0] : [0, score(binaryHead)];
    const probabilities = softmax(logits);
    const selectedLogit = logits[selectedClass-1];
    window.drawClassificationNetwork(get("class-network"),
      {family, classIndex:selectedClass, coordinate:j, activations, probabilities}, selectClass, selectCoordinate);
    for (const button of document.querySelectorAll("[data-outcome]")) {
      button.setAttribute("aria-pressed", button.dataset.outcome === family);
    }
    for (const box of document.querySelectorAll("[data-family]")) box.hidden = box.dataset.family !== family;
    get("class-select-row").hidden = !categorical;
    classSelect.value = classIndex;
    coordinateSelect.value = j;
    coordinateSelect.disabled = reference;
    get("class-coordinate-row").hidden = reference;
    get("class-weight").disabled = reference;
    get("class-edit").hidden = reference;
    get("class-reference").hidden = !reference;
    if (!reference) {
      const weight = (categorical ? categoricalHeads[classIndex-1] : binaryHead)[j-1];
      const term = categorical ? "categorical" : j === 9 ? "binary-bias" : "binary-weight";
      for (const span of document.querySelectorAll("[data-term]")) {
        span.hidden = span.dataset.term !== term;
        for (const number of span.querySelectorAll("[data-selection]")) {
          number.textContent = number.dataset.selection === "head" ? classIndex : j;
        }
        for (const math of span.querySelectorAll("[data-selection-tex]")) {
          math.setAttribute("alttext", math.dataset.selectionTex.replaceAll("[[head]]",classIndex).replaceAll("[[edge]]",j));
        }
      }
      get("class-weight").value = weight;
      get("class-weight-value").value = weight.toFixed(1);
      const source = j === 9 ? "Bias input 1" : "Activation " + activations[j-1];
      const contribution = Number((weight * activations[j-1]).toFixed(3));
      get("class-contribution").textContent = source + " × weight " + weight + " = " + contribution +
        (categorical ? " in the logit for class " + classIndex + "." : " in the event log odds.");
    }
    get("class-logit-label").textContent = categorical ? "Logit for class " + classIndex : "Log odds of Y = 1";
    get("class-logit-value").textContent = Number(selectedLogit.toFixed(3));
    displayProbabilities(probabilities, selectedClass);
  }
  for (const button of document.querySelectorAll("[data-outcome]")) {
    button.addEventListener("click", () => { family = button.dataset.outcome; render(); });
  }
  classSelect.addEventListener("change", () => selectClass(Number(classSelect.value)));
  coordinateSelect.addEventListener("change", () => selectCoordinate(Number(coordinateSelect.value)));
  get("class-weight").addEventListener("input", e => {
    const head = family === "binary" ? binaryHead : categoricalHeads[classIndex-1];
    if (!head) return;
    head[coordinates[family]-1] = e.target.valueAsNumber;
    render();
  });
  window.renderOutcomeExplorer = render;
})();
