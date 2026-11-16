"use strict";

const $ = id => document.getElementById(id);
const fmt = (x, digits = 4) => Number(x.toPrecision(digits)).toString();
const volume = n => n.toFixed(2) + " ft³";
const colors = {fnn:"#894725", bnn:"#21645d", ink:"#183447", grid:"#cbd1ca", quiet:"#566a71"};
const priors = {
  G:"The diameter weight shares its scale with every other incoming weight and bias in layer one.",
  GL:"The diameter weight has its own local multiplier. Its product with the layer scale gives this weight’s scale.",
  GLT:"The diameter weight has its own scale. The fit uses quadrature to compute its moments and learns a tail shape for the whole layer."
};
let data;
let view = "train";
let trial = 1;
let spreadFactor = 1;
let pass = 0;
let mask = "recorded";
let count = 1;
let timer;
let predictionRange;
const views = {
  principle:"What does regularization change in the objective?",
  train:"What happens to one hidden unit during training?",
  predict:"How is a volume predicted for a new tree?"
};

function make(tag, text) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}
function equationPanel(title, equations, note, result) {
  const block=document.createDocumentFragment();
  block.append(make("h4",title));
  const lines=make("div");
  lines.className="formula-math";
  for (const key of equations) {
    const line=make("div");
    line.innerHTML=regularizationMath[key];
    lines.append(line);
  }
  block.append(lines);
  if (note) {
    const paragraph=make("p",note);
    paragraph.className="formula-note"; block.append(paragraph);
  }
  if (result) {
    const value=make("p",result);
    value.className="formula-result"; block.append(value);
  }
  return block;
}
function showObjectives() {
  const method=$("fnn").value;
  const residual=method==="Dropout" ? "maskedResidual" : "residual";
  const losses={FNN:"leastSquares",L1:"penalizedLoss",Dropout:"dropoutLoss"};
  const notes={
    FNN:"Each residual is a training tree’s standardized volume minus the network prediction.",
    L1:"The penalty sums over all weights and biases. Each residual is measured on the standardized training scale.",
    Dropout:"For each training tree, the expectation averages over the masks on all hidden units."
  };
  $("fnn-objective").replaceChildren(equationPanel("Minimize the training loss", [residual,losses[method]],notes[method]));
  $("bnn-objective").replaceChildren(equationPanel("Maximize the evidence lower bound",["elbo"],
    "The likelihood measures fit to the training data. The joint prior acts on all parameters. The Jeffreys variance prior fixes this objective only up to an additive constant."));
}
function showValues(id, rows) {
  $(id).replaceChildren(...rows.map(([label,value]) => {
    const row=make("div");
    row.append(make("dt",label),make("dd",value)); return row;
  }));
}
function stop() {
  clearInterval(timer);
  timer=undefined;
  if ($("animate")) $("animate").textContent="Play draws";
}
function slider(id,title,min,max,increment,value,change) {
  const label=make("label",title), output=make("output");
  label.htmlFor=id; output.id=id+"-value"; output.setAttribute("for",id); label.append(output);
  const input=make("input"); input.type="range"; input.id=id; input.min=min; input.max=max;
  input.step=increment; input.value=value;
  input.addEventListener("input",() => {stop(); change(Number(input.value)); render();});
  $("controls").append(label,input);
}
function setControls() {
  const controls=$("controls"); controls.replaceChildren();
  if (view==="principle") {
    slider("trial","Trial weight or mean",-2.5,2.5,.05,trial,v=>trial=v);
    slider("spread","Trial SD / fitted SD",.25,2,.05,spreadFactor,v=>spreadFactor=v);
  } else if (view==="train") {
    slider("pass","Bayesian draw in the first gradient estimate",1,4,1,pass+1,v=>pass=v-1);
    if ($("fnn").value==="Dropout") {
      const label=make("label","Dropout mask"), select=make("select");
      select.id="mask"; label.htmlFor="mask";
      for (const [value,text] of [["recorded","Recorded training pass"],["keep","What if the unit were kept?"],["drop","What if the unit were dropped?"]]) {
        const opt=make("option",text); opt.value=value; select.append(opt);
      }
      select.value=mask;
      select.addEventListener("change",()=>{mask=select.value;render();});
      controls.append(label,select);
    }
  } else {
    slider("count","Weight draws in the Bayesian mean (S)",1,data.control.prediction_draws,1,count,v=>count=v);
    const animate=make("button","Play draws"); animate.type="button"; animate.id="animate";
    animate.addEventListener("click",()=>{
      if (timer!==undefined) {stop();return;}
      if (count===data.control.prediction_draws) count=1;
      animate.textContent="Pause draws"; render();
      timer=setInterval(()=>{
        count=Math.min(data.control.prediction_draws,count+(count<16 ? 1 : 5));
        if (count===data.control.prediction_draws) stop();
        render();
      },120);
    });
    controls.append(animate);
  }
}

function principle(kind, key, method) {
  const bayes=kind==="bnn", penalized=key==="L1";
  const etaReg=data.l1.coefficient, omega=method.effective_precision_after;
  const scale=bayes ? method.sd_after*spreadFactor : undefined;
  const cost=t=>bayes ? omega*(t*t+scale*scale)/2 : penalized ? etaReg*Math.abs(t) : 0;
  const slope=bayes ? omega*trial : penalized ? etaReg*Math.sign(trial) : 0;
  let formula, description, summary;
  if (bayes) {
    summary="The diameter weight has a Gaussian factor with mean m and standard deviation s.";
    formula=equationPanel("Expected cost from the weight prior",["priorCost","precision"],
      "Here λ is this weight’s shrinkage scale and σ² is the observation variance.");
    description=priors[key]+" Increasing the trial standard deviation raises both this cost and the Gaussian entropy. The ELBO accounts for both.";
  } else if (penalized) {
    summary="The fit minimizes half the mean squared training error plus an L₁ penalty on the weights and biases.";
    formula=equationPanel("Penalty on the diameter weight",["l1Penalty"],"The penalty strength is 0.001.");
    description="The penalty adds a positive slope above zero and a negative slope below zero. At zero the implementation chooses subgradient zero.";
  } else if (key==="Dropout") {
    summary="Dropout trains one set of weights by averaging squared error over random hidden-unit masks.";
    formula=equationPanel("Penalty on the diameter weight",["noPenalty"],"Dropout acts on the hidden activations during training.");
    description="Changing a mask changes the network prediction and its gradient. The training-pass tab follows one such mask.";
  } else {
    summary="The fit minimizes half the mean squared training error using one set of weights and biases.";
    formula=equationPanel("Penalty on the diameter weight",["noPenalty"],"The unpenalized fit assigns zero penalty at every weight value.");
    description="The weight still receives a gradient from the squared prediction errors on the training trees.";
  }
  const rows=bayes ? [["Scale precision P",fmt(method.scale_precision_after)],["Noise precision",fmt(method.noise_precision_after)],["Weight precision ω",fmt(omega)],["Trial standard deviation s",fmt(scale)],["Entropy change from fitted spread",fmt(Math.log(spreadFactor))+" nats"]] :
                    [["Fitted diameter weight",fmt(method.weight_after)]];
  rows.push(["Local cost at the trial value",fmt(cost(trial))],
            [bayes ? "Cost derivative with respect to m" : "Penalty subgradient at w",fmt(slope)]);
  return {formula,description,summary,rows,cost,
    caption:bayes ? "The point gives the expected negative log-prior contribution at the trial mean, up to a constant. The curve holds the trial SD and the scale and noise factors fixed." :
                   "The point gives this coefficient’s penalty at the trial weight. The prediction error enters the training objective separately."};
}

function training(kind,key,method) {
  const bayes=kind==="bnn", drops=key==="Dropout";
  const recorded=method.training_pass[bayes ? pass : 0];
  let multiplier=recorded.multiplier;
  if (drops && mask!=="recorded") multiplier=mask==="keep" ? data.dropout.retained_multiplier : 0;
  const passed=recorded.activation*multiplier;
  const caption = bayes ? "The two inputs and the bias give the weighted sum z in this sampled network. The unit passes its full ReLU activation to layer two." :
      drops ? "The mask acts after ReLU. The two incoming weights and bias still exist when the unit sends zero." :
              "The hidden unit sends its full activation to layer two using the common starting weights.";
  let formula,description,summary;
  if (bayes) {
    summary="Sample the weights and biases, then pass the inputs through that network.";
    formula=equationPanel("Sample the diameter weight",["draw","relu"],"The sampled weights and bias determine the unit’s weighted sum z.");
    description="The three priors start from the same Gaussian weight factors and four antithetic network draws. Backpropagation estimates the likelihood gradient. Adam combines it with gradients from the prior and the Gaussian entropy to update the means and log standard deviations.";
  } else if (drops) {
    summary="Draw a mask for this tree after computing the unit’s ReLU activation.";
    formula=equationPanel("Pass the masked activation",["relu","dropout"],"The mask M is 1 with probability 0.8 and 0 otherwise.");
    description=mask==="recorded" ? "The recorded mask on this training tree sets the activation to zero. Use the what-if control to keep this very same unit. A kept unit sends 1.25 times its activation." :
      "This is a counterfactual mask on the same tree and weights. The recorded first pass dropped the unit. Retention has probability 0.8 during training.";
  } else if (key==="L1") {
    summary="Compute the activation at the current weights. The penalty enters the backward pass.";
    formula=equationPanel("Pass the full activation",["relu"],"Here z is the weighted sum of the two inputs plus the bias.");
    description="The forward computation matches the unpenalized network at the common start. The difference enters in backpropagation, when the penalty gradient is added before the Adam step.";
  } else {
    summary="Compute the activation at the current weights, then update the weights using the training errors.";
    formula=equationPanel("Pass the full activation",["relu"],"Here z is the weighted sum of the two inputs plus the bias.");
    description="A forward pass computes the training prediction at one weight vector. Backpropagation differentiates the squared-error loss and Adam updates that vector.";
  }
  const rows=[["Starting diameter "+(bayes ? "weight mean" : "weight"),fmt(method.weight_before)]];
  if (bayes) rows.push(["Gaussian perturbation ε",fmt(recorded.epsilon)],["Sampled diameter weight",fmt(recorded.weight)],["Starting weight precision ω",fmt(method.effective_precision_before)]);
  if (key==="L1") rows.push(["Penalty gradient at this weight",fmt(method.penalty_subgradient_before)]);
  rows.push(["ReLU activation before masking",fmt(recorded.activation)],["Activation sent to the next layer",fmt(passed)]);
  return {formula,summary,description,rows,caption,recorded,multiplier,passed};
}

function prediction(kind,key,method) {
  const bayes=kind==="bnn", draws=method.prediction.draws.slice(0,count);
  const mean=bayes ? draws.reduce((s,d)=>s+d.volume,0)/count : method.prediction.mean;
  const latest=bayes ? draws[count-1] : null;
  const rows=bayes ? [["Fitted diameter weight, mean ± SD",fmt(method.weight_after)+" ± "+fmt(method.sd_after)],["Latest sampled diameter weight",fmt(latest.weight)],["Latest sampled mean volume",volume(latest.volume)],
                       ["Average of "+count+(count===1 ? " network" : " networks"),volume(mean)],["Full 256-draw average",volume(method.prediction.mean)]] :
                     [["Fitted diameter weight",fmt(method.weight_after)],["Prediction from one fitted network",volume(mean)]];
  const scaling=data.response_scaling;
  const standardized=(mean-scaling.mean)/scaling.sd;
  const calculation=scaling.mean.toFixed(2)+" + "+scaling.sd.toFixed(3)+" × "+standardized.toFixed(4)+" ≈ "+volume(mean);
  const summary=bayes ? "Draw S networks from the fitted weight distribution and average their predictions for this tree." :
      "Use the fitted weights and biases to predict once at this tree’s standardized diameter and height.";
  const formula=equationPanel("Predicted mean volume", [bayes ? "bnnVolume" : "fnnVolume"], null,calculation);
  const description=bayes ? "Each draw samples every weight and bias and gives one conditional mean volume. Averaging these volumes estimates the expectation under the fitted Gaussian weight factors. These draws exclude observation noise." :
      key==="Dropout" ? "All hidden units operate with the fitted weights at prediction. Inverted scaling during training lets the benchmark use the ordinary network here." :
      "The same fitted network gives the same predicted mean each time it is evaluated at this tree.";
  return {mean,draws,formula,summary,description,rows,
    caption:bayes ? "Each dot is a sampled network’s mean volume. The diamond marks the average of the selected draws." :
                   "The diamond marks the fitted network’s mean volume. Both charts use the same volume scale."};
}

function chart(id,kind,result) {
  const canvas=$(id), ctx=canvas.getContext("2d");
  const W=canvas.clientWidth,H=canvas.clientHeight,ratio=window.devicePixelRatio||1;
  if (!W || !H) return;
  canvas.width=Math.round(W*ratio); canvas.height=Math.round(H*ratio);ctx.scale(ratio,ratio);
  const accent=colors[kind], left=48, right=W-20, top=22, bottom=H-40;
  function line(x1,y1,x2,y2,ink,width=1) {
    ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.strokeStyle=ink;ctx.lineWidth=width;ctx.stroke();
  }
  function text(s,x,y,align="center",ink=colors.ink,bold=false) {
    ctx.fillStyle=ink;ctx.textAlign=align;ctx.font=(bold?"600 ":"")+'12px "Segoe UI", sans-serif';ctx.fillText(s,x,y);
  }
  function dot(x,y,r,ink) {ctx.fillStyle=ink;ctx.beginPath();ctx.arc(x,y,r,0,2*Math.PI);ctx.fill();}
  if (view==="principle") {
    const x=w=>left+(right-left)*(w+2.5)/5;
    const max=result.cost(2.5);
    const y=value=>max ? bottom-(bottom-top)*value/max : bottom;
    for(const v of (max ? [0,max/2,max] : [0])) {
      line(left,y(v),right,y(v),colors.grid);text(fmt(v,2),left-6,y(v)+4,"right",colors.quiet);
    }
    for(const t of [-2,0,2]) text(String(t),x(t),bottom+16);
    ctx.beginPath();
    for(let i=0;i<=100;i++) {
      const t=-2.5+i/20;
      if(i===0) ctx.moveTo(x(t),y(result.cost(t))); else ctx.lineTo(x(t),y(result.cost(t)));
    }
    ctx.strokeStyle=accent;ctx.lineWidth=2.5;ctx.stroke();
    dot(x(trial),y(result.cost(trial)),5,accent);
    text(kind==="bnn"?"Trial Gaussian mean m":"Trial weight w",(left+right)/2,H-4);
  } else if (view==="train") {
    const center=W*.52, leftNode=W*.21, rightNode=W*.86, mid=H*.53;
    const inputs=[["Diameter",-30],["Height",0],["1",30]];
    for (const [label,dy] of inputs) {
      const color=dy<0 ? accent : colors.quiet;
      line(leftNode+8,mid+dy,center-15,mid,color,dy<0 ? 2.8 : 1.4);
      dot(leftNode,mid+dy,dy<0 ? 7 : 5,color);
      text(label,leftNode-10,mid+dy+4,"right");
    }
    const end=rightNode-13;
    line(center+17,mid,end,mid,result.multiplier===0 ? colors.grid : accent,2.8);
    if (result.multiplier===0) {
      const cross=(center+rightNode)/2;
      line(cross-5,mid-6,cross+5,mid+6,accent,2);line(cross-5,mid+6,cross+5,mid-6,accent,2);
    } else {
      line(end-7,mid-4,end,mid,accent,2);line(end-7,mid+4,end,mid,accent,2);
    }
    dot(center,mid,16,accent);dot(rightNode,mid,10,result.multiplier===0 ? colors.quiet : accent);
    text("w = "+fmt(result.recorded.weight),W*.35,mid-35);
    text("bias",W*.35,mid+37,"center",colors.quiet);
    text("ReLU",center,mid+4,"center","white",true);
    text("× "+fmt(result.multiplier),W*.69,mid-9,"center",accent,true);
    text("Layer 2",rightNode,mid-23);text(fmt(result.passed),rightNode,mid+34,"center",accent,true);
  } else {
    const [lo,hi]=predictionRange, x=v=>left+(right-left)*(v-lo)/(hi-lo);
    const observed=x(data.tree.observed_volume);
    line(observed,top,observed,bottom,colors.quiet,1.3);
    text("Observed "+data.tree.observed_volume+" ft³",observed,top-6,"center",colors.quiet);
    line(left,bottom,right,bottom,colors.grid);
    for(let v=lo;v<=hi;v+=(hi-lo)/4) text(fmt(v,3),x(v),bottom+16);
    if(kind==="bnn") {
      // Separate coincident outputs vertically within narrow volume bins.
      const bins=new Map();
      ctx.globalAlpha=.6;
      for(const draw of result.draws) {
        const px=x(draw.volume), bin=Math.floor((px-left)/5), index=bins.get(bin)||0;
        bins.set(bin,index+1);
      }
      const spacing=Math.min(5,(bottom-top-16)/Math.max(1,...bins.values()));
      bins.clear();
      for(const draw of result.draws) {
        const px=x(draw.volume), bin=Math.floor((px-left)/5), index=bins.get(bin)||0;
        dot(px,bottom-9-index*spacing,2.2,accent);bins.set(bin,index+1);
      }
      ctx.globalAlpha=1;
    }
    const px=x(result.mean), py=bottom;
    ctx.fillStyle=accent;ctx.beginPath();ctx.moveTo(px,py-7);ctx.lineTo(px+7,py);
    ctx.lineTo(px,py+7);ctx.lineTo(px-7,py);ctx.closePath();ctx.fill();
    text("Timber volume (ft³)",(left+right)/2,H-4);
  }
}

function render() {
  $("question").textContent=views[view];
  if(view==="principle") {
    $("trial-value").textContent=trial.toFixed(2);
    $("spread-value").textContent=spreadFactor.toFixed(2)+"×";
    $("instructions").textContent="Move the slider to compare the direct cost of one diameter weight. The frequentist curve shows its explicit penalty. The Bayesian curve shows its expected cost under the weight prior, with the noise and shrinkage factors fixed. Each chart uses its own vertical scale.";
  } else if(view==="train") {
    $("pass-value").textContent=(pass+1)+" of 4";
    const t=data.training_tree;
    $("instructions").textContent="Training tree "+t.row+" has diameter "+t.diameter+" in and height "+t.height+" ft. These enter the network as "+fmt(t.standardized_input[0],3)+" and "+fmt(t.standardized_input[1],3)+" after standardization. Compare one frequentist forward pass with the four draws used for the first Bayesian gradient.";
  } else {
    $("count").value=count;$("count-value").textContent=count+" of "+data.control.prediction_draws;
    const t=data.tree;
    const scaling=data.response_scaling;
    $("instructions").replaceChildren(
      "Held-out tree "+t.row+" had a diameter of "+t.diameter+" in and a height of "+t.height+" ft. Their standardized values form ",
      make("b","x"),make("sub","∗"),". Its observed volume was "+t.observed_volume+" ft³. The training mean was ",
      make("var","ȳ")," = "+scaling.mean.toFixed(2)+" ft³ and the sample standard deviation was ",
      make("var","s"),make("sub","y")," = "+scaling.sd.toFixed(3)+" ft³."
    );
  }
  for (const kind of ["fnn","bnn"]) {
    const key=$(kind).value, method=data.methods[key];
    const result=view==="principle" ? principle(kind,key,method) : view==="train" ? training(kind,key,method) : prediction(kind,key,method);
    $(kind+"-summary").textContent=result.summary;
    $(kind+"-formula").replaceChildren(result.formula);
    $(kind+"-description").textContent=result.description;
    $(kind+"-caption").textContent=result.caption;
    showValues(kind+"-values",result.rows); chart(kind+"-plot",kind,result);
  }
  $("bridge").hidden=view!=="principle";
  if (view==="principle" && $("objectives").open) showObjectives();
}

function changeView(next) {
  stop();view=next;
  document.querySelectorAll("[data-view]").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.view===view)));
  setControls();render();
}
async function init() {
  $("status").hidden=false;
  const response=await fetch("../data/regularization-example.json");
  if(!response.ok) throw new Error("Missing example");
  data=await response.json();
  const volumes=[data.tree.observed_volume];
  for(const method of Object.values(data.methods)) {
    volumes.push(method.prediction.mean,...method.prediction.draws.map(d=>d.volume));
  }
  predictionRange=[Math.floor((Math.min(...volumes)-.5)/2)*2,Math.ceil((Math.max(...volumes)+.5)/2)*2];
  $("workbench").hidden=false;$("status").hidden=true;
  $("objectives").addEventListener("toggle",()=>{if ($("objectives").open) showObjectives();});
  document.querySelectorAll("[data-view]").forEach(b=>b.addEventListener("click",()=>changeView(b.dataset.view)));
  for(const kind of ["fnn","bnn"]) $(kind).addEventListener("change",()=>{stop();setControls();render();});
  document.addEventListener("visibilitychange",()=>{if(document.hidden)stop();});
  changeView(view);
  const observer=new ResizeObserver(()=>render());
  observer.observe($("fnn-plot")); observer.observe($("bnn-plot"));
}
init().catch(()=>{
  $("workbench").hidden=true;$("status").hidden=false;
  $("status").textContent="The recorded example could not be loaded. The companion notes under Reading and R code describe all six methods.";
});
