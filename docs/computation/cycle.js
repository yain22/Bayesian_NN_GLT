"use strict";

const element = id => document.getElementById(id);
const priorButtons = Array.from(document.querySelectorAll("[data-prior]"));
const stepButtons = Array.from(document.querySelectorAll("[data-step]"));
const color = {old: "#728692", fit: "#176962", ink: "#183447", rule: "#d7dad3"};
const format = n => Number(n.toPrecision(5)).toString();
let fixture;
let prior = "GLT";
let step = 0;
let timer;

function stageContent() {
  const {before, after, update} = fixture.priors[prior];
  const u = update.scale_update;
  const noise = after.noise_precision;
  switch (step) {
    case 0:
      return {
        title: "Fit the Gaussian weight factor", plot: "The diameter weight",
        text: "Each draw gives a possible network and a likelihood gradient. The weight block updates its mean and log standard deviation while the current prior precision stays fixed.",
        formula: "draw",
        rows: [["Mean, before → after", format(before.mean)+" → "+format(after.mean)],
               ["Standard deviation, before → after", format(before.sd)+" → "+format(after.sd)],
               ["Accepted Adam proposals", update.accepted_steps+" of "+fixture.control.weight_steps]],
        caption: "Gray shows the initial Gaussian factor. Green shows the factor after the weight block. The horizontal axis is the standardized network coefficient."
      };
    case 1:
      return {
        title: "Carry forward an expected square", plot: "Two parts of the second moment",
        text: "The scale update needs the expected squared weight. The Gaussian factor supplies this exactly, including the contribution from its remaining uncertainty.",
        formula: "moment",
        rows: [["Squared mean, m²", format(after.mean**2)], ["Variance, s²", format(after.sd**2)],
               ["Expected square, H", format(after.second_moment)]],
        caption: "The width of each segment is its share of the expected square. Each hidden weight and bias supplies a moment of its own."
      };
    case 2:
      return {
        title: "Update the Gaussian noise variance", plot: "What enters the noise update",
        text: "Use fresh weight draws for the expected residual sum of squares. Add the new hidden-weight moments weighted by the old scale precisions. In this Jeffreys-prior fit, half that total gives the inverse-gamma scale b.",
        formula: "noise",
        rows: [["Residual sum of squares ("+fixture.control.noise_draws+" draws)", format(update.expected_sse)],
               ["Hidden-weight contribution", format(update.hidden_penalty)],
               ["Shape a / scale b", format(update.noise_shape)+" / "+format(update.noise_scale)],
               ["Updated noise precision", format(noise)]],
        caption: "The two segments sum to twice b. With 25 training trees and 17 hidden coefficients (including biases), the shape is (25 + 17)/2 = 21."
      };
    case 3: {
      const content = {
        plot: "Scale precision for this connection",
        caption: "Expected inverse squared scale after one cycle for the same connection under each prior. The axis uses a logarithmic scale.",
        rows: []
      };
      if (prior === "G") {
        content.title = "Pool the first layer’s weights";
        content.text = "G assigns one scale to all nine incoming weights and biases in layer one. Their expected squares enter the inverse-gamma layer update. The resulting scale precision is shared by all nine.";
        content.formula = "global";
        content.rows = [["Sum of the layer’s expected squares",format(update.layer_second_moment_sum)],
                        ["Shared scale precision, P",format(after.scale_precision)]];
      } else if (prior === "GL") {
        content.title = "Combine the local and layer scales";
        content.text = "GL updates a squared local multiplier and its auxiliary for each coefficient, then the squared layer scale and its auxiliary. The two inverse moments multiply because their variational factors are independent.";
        content.formula = "horseshoe";
        content.rows = [["Layer inverse moment",format(u.layer_inverse_squared_scale)],
                        ["Local inverse moment",format(u.multiplier_inverse_squared_scale)],
                        ["Scale precision, P",format(after.scale_precision)]];
      } else {
        content.title = "Integrate the GLT local scale";
        content.text = "For this connection, the new noise precision times H gives B. The current layer and mixing factors give A. Quadrature gives the expected local scale. Multiply this mean by A/B to obtain the scale precision P. The layer, auxiliary and log-shape updates follow.";
        content.formula = "tail";
        content.rows = [["A",format(u.kernel_A)], ["B",format(u.kernel_B)], ["Expected local scale",format(u.mean_local_scale)],
                        ["Scale precision, P",format(after.scale_precision)]];
      }
      return content;
    }
    case 4:
      return {
        title: "Return the precision to the weights", plot: "Shrinkage in the next weight block",
        text: "Multiply the updated noise and scale precisions. In the next block the prior contributes −ωm to the gradient for this weight’s mean. The data contribute their own likelihood gradient.",
        formula: "return",
        rows: [["Noise × scale precision",format(noise)+" × "+format(after.scale_precision)],
               ["Effective precision, before → after",format(before.effective_precision)+" → "+format(after.effective_precision)],
               ["Next prior gradient at the current m",format(-after.effective_precision*after.mean)]],
        caption: "The curves show the penalty ωm²/2 as the mean varies at fixed variance. Gray uses the starting precision and green uses the precision handed to the next block."
      };
  }
}

function drawPlot() {
  if (!fixture) return;
  const canvas = element("plot");
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (!width || !height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const left = 46, right = width - 18, top = 40, bottom = height - 44;
  const {before, after, update} = fixture.priors[prior];
  function label(value, x, y, align = "left", ink = color.ink) {
    ctx.font = '12px "Segoe UI", sans-serif';
    ctx.fillStyle = ink;
    ctx.textAlign = align;
    ctx.fillText(value, x, y);
  }
  function line(x1,y1,x2,y2,ink,weight=1) {
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2);
    ctx.strokeStyle = ink; ctx.lineWidth = weight; ctx.stroke();
  }
  if (step === 0 || step === 4) {
    const gaussian = step === 0;
    const bound = Math.max(Math.abs(before.mean)+3*before.sd, Math.abs(after.mean)+3*after.sd);
    const lo = gaussian ? Math.min(before.mean-3.5*before.sd, after.mean-3.5*after.sd) : -bound;
    const hi = gaussian ? Math.max(before.mean+3.5*before.sd, after.mean+3.5*after.sd) : bound;
    const density = (x, state) => Math.exp(-0.5*((x-state.mean)/state.sd)**2)/(state.sd*Math.sqrt(2*Math.PI));
    const quantity = gaussian ? density : (x,state) => state.effective_precision*x*x/2;
    const ymax = gaussian ? Math.max(density(before.mean,before),density(after.mean,after)) : Math.max(quantity(hi,before),quantity(hi,after));
    const display = v => v===0 ? "0" : Number(v.toPrecision(2)).toString();
    for (let i = 0; i <= 2; i++) {
      const y = bottom - (bottom-top)*i/2;
      line(left,y,right,y,color.rule); label(display(ymax*i/2),left-6,y+4,"right");
    }
    for (let i = 0; i <= 4; i++) {
      const x = left+(right-left)*i/4;
      label(display(lo+(hi-lo)*i/4),x,bottom+18,"center");
    }
    label(gaussian ? "Density" : "Penalty",left,top-16);
    label(gaussian ? "Weight" : "Weight mean", (left+right)/2, height-4, "center");
    for (const [state,ink] of [[before,color.old],[after,color.fit]]) {
      ctx.beginPath();
      for (let i = 0; i <= 160; i++) {
        const t = i/160, x = left+(right-left)*t;
        const y = bottom-(bottom-top)*quantity(lo+(hi-lo)*t,state)/ymax;
        if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
      ctx.lineWidth=2.5; ctx.strokeStyle=ink; ctx.stroke();
    }
    return;
  }
  if (step === 3) {
    const xs = Object.values(fixture.priors).map(d => d.after.scale_precision);
    const min = Math.min(0,Math.floor(Math.log10(Math.min(...xs))));
    const max = Math.max(min+1,Math.ceil(Math.log10(Math.max(...xs))));
    const pos = v => left+(right-left)*(Math.log10(v)-min)/(max-min);
    for (let exponent=min; exponent<=max; exponent++) {
      const x=pos(10**exponent);
      line(x,top-5,x,bottom,color.rule);
      label(String(10**exponent),x,bottom+18,"center");
    }
    Object.entries(fixture.priors).forEach(([key,d],i) => {
      const y=top+16+i*(bottom-top-24)/2, x=pos(d.after.scale_precision);
      const ink=key===prior ? color.fit : color.old;
      label(key,left-9,y+4,"right",ink); line(left,y,x,y,ink,2);
      ctx.beginPath();ctx.arc(x,y,key===prior ? 6 : 4,0,2*Math.PI);ctx.fillStyle=ink;ctx.fill();
    });
    label("Expected inverse squared scale",(left+right)/2,height-4,"center");
    return;
  }
  const moments = step === 1;
  const values = moments ? [after.mean**2,after.sd**2] : [update.expected_sse,update.hidden_penalty];
  const labels = moments ? ["Squared mean","Variance"] : ["Residuals","Hidden weights"];
  const share = values[0]/(values[0]+values[1]), x0 = 8, span = width-16, y = height*.36;
  ctx.fillStyle=color.fit; ctx.fillRect(x0,y,span*share,35);
  ctx.fillStyle="#ad6545"; ctx.fillRect(x0+span*share,y,span*(1-share),35);
  labels.forEach((name,i) => {
    const py=y+65+i*23;
    ctx.fillStyle=i===0 ? color.fit : "#ad6545"; ctx.fillRect(x0,py-9,9,9);
    label(name,x0+17,py);label(format(values[i]),width-8,py,"right");
  });
  label(moments ? "Expected squared weight" : "Contributions to the variance update",x0,y-14);
}

function stopPlayback() {
  clearInterval(timer);
  timer = undefined;
  element("play").textContent = "Play cycle";
}

function render() {
  const content = stageContent();
  element("stage-title").textContent = content.title;
  element("plot-title").textContent = content.plot;
  element("plot-caption").textContent = content.caption;
  element("explanation").textContent = content.text;
  element("equation").replaceChildren(element("formula-" + content.formula).content.cloneNode(true));
  element("position").textContent = prior+" · Step "+(step+1)+" of 5";
  element("announce").textContent = element("position").textContent+". "+content.title;
  element("readout").replaceChildren(...content.rows.map(([name,value]) => {
    const row=document.createElement("div"), term=document.createElement("dt"), result=document.createElement("dd");
    term.textContent=name; result.textContent=value; row.append(term,result); return row;
  }));
  priorButtons.forEach(b => b.setAttribute("aria-pressed", String(b.dataset.prior === prior)));
  stepButtons.forEach((b,i) => {
    if (i === step) b.setAttribute("aria-current","step"); else b.removeAttribute("aria-current");
  });
  element("back").disabled = step === 0;
  element("next").disabled = step === 4;
  drawPlot();
}

function goTo(index) {
  stopPlayback();
  step = index;
  render();
}
async function initialize() {
  const response = await fetch("../data/trees-cycle.json");
  if (!response.ok) throw new Error("Example unavailable");
  fixture = await response.json();
  element("replay").hidden = false;
  priorButtons.forEach(b => b.addEventListener("click",() => {prior=b.dataset.prior; goTo(step);}));
  stepButtons.forEach((b,i) => b.addEventListener("click",() => goTo(i)));
  element("back").addEventListener("click",() => goTo(Math.max(0,step-1)));
  element("next").addEventListener("click",() => goTo(Math.min(4,step+1)));
  element("play").addEventListener("click",() => {
    if (timer !== undefined) {stopPlayback(); return;}
    if (step === 4) goTo(0);
    element("play").textContent="Pause";
    timer=setInterval(() => {
      step++;
      if (step===4) stopPlayback();
      render();
    },5500);
  });
  document.addEventListener("visibilitychange",() => {if (document.hidden) stopPlayback();});
  new ResizeObserver(drawPlot).observe(element("plot"));
  render();
}
initialize().catch(() => {
  element("replay").hidden=true;
  element("load-status").hidden=false;
  element("load-status").textContent="The recorded values could not be loaded. The written guide below explains the full cycle.";
});
