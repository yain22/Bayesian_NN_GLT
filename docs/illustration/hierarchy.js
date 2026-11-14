/* Nodes outside the plate are shared by the layer; nodes inside repeat over individual coefficients. */
(() => {
  "use strict";
  const ns = "http://www.w3.org/2000/svg";
  const graphs = {
    "G:original": {nodes:{fixedScale:[36,76],tau:[126,76],w:[289,76]},edges:[["fixedScale","tau"],["tau","w"]]},
    "G:expanded": {nodes:{fixedIG:[25,76],chiTau:[90,76],t:[158,76],w:[289,76]},edges:[["fixedIG","chiTau"],["chiTau","t"],["t","w"]]},
    "GL:original": {nodes:{fixedScale:[26,86],tau:[127,86],r:[221,37],lambda:[221,86],w:[304,86]},edges:[["fixedScale","tau"],["tau","lambda"],["r","lambda"],["lambda","w"]]},
    "GL:expanded": {nodes:{fixedIG:[20,89],chiTau:[79,89],t:[144,89],chiR:[211,40],tr:[300,40],w:[300,89]},edges:[["fixedIG","chiTau"],["chiTau","t"],["chiR","tr"],["tr","w"],["t","w"]]},
    "GLT:original": {nodes:{hyper:[34,36],xi:[129,36],K:[35,91],tau:[129,91],lambda:[222,91],w:[304,91]},edges:[["hyper","xi"],["xi","tau"],["K","tau"],["xi","lambda"],["tau","lambda"],["lambda","w"]]},
    "GLT:expanded": {nodes:{hyper:[31,37],xi:[121,37],K:[30,91],tau:[121,91],nu:[219,37],lambda:[219,91],w:[301,91]},edges:[["hyper","xi"],["xi","tau"],["K","tau"],["xi","nu"],["tau","lambda"],["nu","lambda"],["lambda","w"]]}
  };
  function el(tag,attrs={},txt) {
    const node=document.createElementNS(ns,tag);
    for(const [k,v] of Object.entries(attrs)) node.setAttribute(k,v);
    if(txt!==undefined) node.textContent=txt;
    return node;
  }
  window.drawHierarchy = function(svg,prior,form,layer,unit,coordinate,dimension,width) {
    const {nodes,edges}=graphs[prior+":"+form];
    const l = String(layer), kl = String(unit) + l, klj = kl + "," + coordinate;
    const labels = {fixedScale:["1/M",l], fixedIG:["M",l], tau:["τ",l], r:["r",klj],
      w:coordinate === dimension ? ["b",kl] : ["w",klj], lambda:["λ",klj], xi:["ξ",l],
      nu:["ν",klj], t:["t",l], chiTau:["χ","τ,"+l], tr:["t","r,"+klj], chiR:["χ","r,"+klj],
      K:["M",l], hyper:["μξ, ρ²ξ",""]};
    svg.setAttribute("viewBox","0 0 345 142");
    svg.setAttribute("role","img");
    svg.setAttribute("aria-labelledby","graph-title graph-description");
    const descriptions = {fixedScale:"fixed prior scale 1 divided by M in layer "+l,
      fixedIG:"fixed squared coefficient count M squared in layer "+l,
      chiTau:"random layer auxiliary chi, tau, "+l,
      chiR:"random coefficient auxiliary chi, r, "+klj};
    const describe = name=>descriptions[name] || labels[name].join("_");
    const def=el("defs"),marker=el("marker",{id:"prior-arrow",viewBox:"0 0 8 8",refX:7,refY:4,markerWidth:6,markerHeight:6,orient:"auto-start-reverse"});
    marker.append(el("path",{d:"M0 0 L8 4 L0 8 Z",fill:"#667e83"}));def.append(marker);
    svg.replaceChildren(el("title",{id:"graph-title"},prior+"-BNN "+form+" hierarchy for coefficient "+coordinate+" of unit "+unit+" in layer "+layer),el("desc",{id:"graph-description"},
      "Conditional on the observation variance sigma squared. "+edges.map(([from,to])=>
        describe(from)+(prior==="GL" && form==="original" && to==="lambda" ? " is a factor in " : " determines the law of ")+describe(to)
      ).join(". ")+". The plate repeats over "+width+" units and "+dimension+" coordinates per unit."),def);
    svg.append(el("rect",{x:185,y:12,width:153,height:121,rx:8,fill:"#f8faf8",stroke:"#a9bcb8","stroke-dasharray":"4 3"}));
    svg.append(el("text",{x:328,y:125,"text-anchor":"end","font-size":11,fill:"#506969"},"k = 1, …, "+width+" · j = 1, …, "+dimension));
    for(const [from,to] of edges){
      const [x,y]=nodes[from],[u,v]=nodes[to],d=Math.hypot(u-x,v-y),dx=(u-x)/d,dy=(v-y)/d;
      svg.append(el("line",{x1:x+20*dx,y1:y+20*dy,x2:u-20*dx,y2:v-20*dy,stroke:"#667e83","stroke-width":1.45,"marker-end":"url(#prior-arrow)"}));
    }
    for(const [key,[x,y]] of Object.entries(nodes)){
      const weight=key==="w",fixed=key==="fixedScale"||key==="fixedIG"||key==="K"||key==="hyper";
      if(!fixed)svg.append(el("circle",{cx:x,cy:y,r:18,fill:weight?"#e4f0ee":"#fffefa",stroke:weight?"#135d7a":"#708387","stroke-width":weight?2.8:1.25}));
      const text=el("text",{x,y:y+4,"text-anchor":"middle",fill:"#213b43","font-family":"Cambria Math, Georgia, serif","font-size":17,"font-style":"italic","font-weight":400},labels[key][0]);
      if(key==="hyper") {
        text.textContent="μ"; text.append(el("tspan",{"baseline-shift":"sub","font-size":10},"ξ"),
          el("tspan",{},", ρ"), el("tspan",{"baseline-shift":"super","font-size":10},"2"),
          el("tspan",{"baseline-shift":"sub","font-size":10},"ξ"));
      } else {
        text.append(el("tspan",{"baseline-shift":"sub","font-size":10,"font-weight":400},labels[key][1]));
        if(key==="fixedIG")text.append(el("tspan",{"baseline-shift":"super","font-size":10},"2"));
      }
      svg.append(text);
    }
  };
})();
