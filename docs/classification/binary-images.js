(() => {
  "use strict";
  const data = window.BINARY_IMAGE_STUDY || window.CIFAR_STUDY;
  if (!data) return;
  const element = id => document.getElementById(id);
  const methods = ["G", "GL", "GLT", "Unpenalized", "Dropout", "L1"];
  const architectures = ["2x10", "2x25", "5x10", "5x25", "10x10", "10x25"];
  const gallery = element("main");
  const side = Number(gallery.dataset.side);
  const area = side * side;
  const channels = element("channel") ? ["red", "green", "blue"] : ["grayscale"];
  const labels = [gallery.dataset.negative, gallery.dataset.positive];
  gallery.style.setProperty("--side", side);
  const fits = new Map(data.fits.map(fit => [fit.depth + "x" + fit.width + "/" + fit.method, fit]));
  const photos = data.images.map(photo => ({
    ...photo, raw: Uint8Array.from(atob(photo.pixels), ch => ch.charCodeAt(0))
  }));
  let selected = 0;
  const option = (select, value, label) => select.add(new Option(label, value));
  const architectureLabel = key => key.replace("x", " × ");
  const methodLabel = method => method === "L1" ? "L₁" : method;
  for (const key of architectures) option(element("architecture"), key, architectureLabel(key));
  for (const method of methods) option(element("model"), method, methodLabel(method));
  element("model").value = "GLT";

  function paint(canvas, raw) {
    const ctx = canvas.getContext("2d");
    const rgba = ctx.createImageData(side, side);
    for (let i = 0; i < area; i++) {
      for (let q = 0; q < 3; q++) rgba.data[4 * i + q] = raw[(channels.length === 1 ? 0 : q) * area + i];
      rgba.data[4 * i + 3] = 255;
    }
    ctx.putImageData(rgba, 0, 0);
  }
  function renderPixel() {
    const row = Number(element("row").value);
    const col = Number(element("column").value);
    if (![row, col].every(i => Number.isInteger(i) && i >= 1 && i <= side)) return;
    const q = element("channel") ? Number(element("channel").value) : 1;
    const start = area * (q - 1) + side * (row - 1);
    const index = start + col - 1;
    const raw = photos[selected].raw;
    const standard = (raw[index] - data.trainingMean[index]) / data.trainingSD[index];
    element("crosshair").style.left = 100 * (col - 1) / side + "%";
    element("crosshair").style.top = 100 * (row - 1) / side + "%";
    let prefix = "";
    if (channels.length === 3) {
      const spatialIndex = side * (row - 1) + col - 1;
      element("channels").textContent = "RGB (" + [0, 1, 2].map(ch => raw[area * ch + spatialIndex]).join(", ") + ")";
      prefix = area.toLocaleString("en-US") + " × " + (q - 1) + " + ";
    }
    element("coordinate").textContent = "j = " + prefix + side + " × " + (row - 1) + " + " + col +
      " = " + (index + 1).toLocaleString("en-US");
    element("intensity").textContent = "Raw " + channels[q - 1] + " intensity " + raw[index] +
      " of 255 · Standardized input " + standard.toFixed(3);
    element("strip-label").textContent = "All " + side + " " + channels[q - 1] + " intensities in row " + row +
      ", from black (0) to white (255). The outline marks column " + col + ".";
    element("strip").replaceChildren(...Array.from({length: side}, (_, c) => {
      const swatch = document.createElement("i");
      const v = raw[start + c];
      swatch.style.backgroundColor = "rgb(" + [v, v, v].join(",") + ")";
      if (c + 1 === col) swatch.className = "selected";
      return swatch;
    }));
  }
  function renderPrediction() {
    const fit = fits.get(element("architecture").value + "/" + element("model").value);
    const photo = photos[selected];
    const probs = fit.probabilities[selected];
    const winner = probs[1] > probs[0] ? 1 : 0;
    element("record").textContent = "Test image #" + photo.id + " · Recorded " +
      labels[photo.truth].toLowerCase() + " · Predicted " + labels[winner].toLowerCase();
    const rows = probs.map((probability, i) => {
      const row = document.createElement("div");
      row.className = "prob-row" + (i === winner ? " winner" : "");
      const name = document.createElement("span");
      name.textContent = labels[i];
      const track = document.createElement("span");
      track.className = "prob-track";
      const fill = document.createElement("span");
      fill.className = "prob-fill";
      fill.style.width = (100 * probability) + "%";
      track.append(fill);
      const value = document.createElement("span");
      value.className = "value";
      value.textContent = (100 * probability).toFixed(1) + "%";
      row.append(name, track, value);
      return row;
    });
    element("probabilities").replaceChildren(...rows);
    element("model-score").textContent = "Across all 2,000 test images this fit classified " +
      fit.correct.toLocaleString("en-US") + " correctly (" + (fit.accuracy * 100).toFixed(2) +
      "%), with mean log loss " + fit.log_loss.toFixed(3) + ".";
  }
  function choosePhoto(index) {
    selected = index;
    Array.from(element("photos").children).forEach((button, i) => button.setAttribute("aria-pressed", String(i === index)));
    paint(element("image"), photos[index].raw);
    element("image").setAttribute("aria-label", "Enlarged " + labels[photos[index].truth] +
      " image, " + gallery.dataset.study + " test image " + photos[index].id);
    renderPixel();
    renderPrediction();
  }
  photos.forEach((photo, i) => {
    const button = document.createElement("button");
    button.className = "photo";
    button.type = "button";
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = side;
    canvas.setAttribute("aria-hidden", "true");
    paint(canvas, photo.raw);
    const label = document.createElement("span");
    label.textContent = labels[photo.truth] + " #" + photo.id;
    button.append(canvas, label);
    button.addEventListener("click", () => choosePhoto(i));
    element("photos").append(button);
  });
  function renderScores() {
    const metric = element("metric").value;
    const accuracy = metric === "accuracy";
    element("table-caption").textContent = accuracy ?
      "Accuracy among 2,000 test images. Bold identifies the highest accuracy in each row." :
      "Mean negative natural log probability assigned to the true class. Bold identifies the lowest loss in each row.";
    const rows = architectures.map(key => {
      const row = document.createElement("tr");
      const heading = document.createElement("th");
      heading.scope = "row";
      heading.textContent = architectureLabel(key);
      row.append(heading);
      const scores = methods.map(method => fits.get(key + "/" + method)[metric]);
      const best = accuracy ? Math.max(...scores) : Math.min(...scores);
      scores.forEach(score => {
        const cell = document.createElement("td");
        const value = accuracy ? (100 * score).toFixed(2) : score.toFixed(3);
        if (score === best) {
          const strong = document.createElement("strong");
          strong.className = "best";
          strong.textContent = value;
          cell.append(strong);
        } else cell.textContent = value;
        row.append(cell);
      });
      return row;
    });
    element("scores").tBodies[0].replaceChildren(...rows);
  }
  element("enlargement").addEventListener("click", event => {
    const rect = element("image").getBoundingClientRect();
    const coordinate = fraction => Math.max(1, Math.min(side, Math.floor(fraction * side) + 1));
    element("row").value = coordinate((event.clientY - rect.top) / rect.height);
    element("column").value = coordinate((event.clientX - rect.left) / rect.width);
    renderPixel();
  });
  for (const id of ["row", "column"]) {
    element(id).addEventListener("input", renderPixel);
    element(id).addEventListener("change", () => {
      const input = element(id);
      input.value = Math.max(1, Math.min(side, Math.round(Number(input.value)) || 1));
      renderPixel();
    });
  }
  element("channel")?.addEventListener("change", renderPixel);
  for (const id of ["model", "architecture"]) element(id).addEventListener("change", renderPrediction);
  element("metric").addEventListener("change", renderScores);
  choosePhoto(0);
  renderScores();
  element("controls").hidden = element("score-panel").hidden = false;
})();
