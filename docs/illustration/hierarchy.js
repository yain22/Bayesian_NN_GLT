/* Nodes outside the plate are shared by the layer; nodes inside repeat over units. */
(() => {
  "use strict";
  const ns = "http://www.w3.org/2000/svg";
  const labels = {a:["a","l"], tau:["τ","l"], w:["w","kl"], lambda:["λ","kl"], xi:["ξ","l"],
    nu:["ν","kl"], t:["t","l"], at:["a","τ,l"], tr:["t","r,kl"], ar:["a","r,kl"], K:["K","l"], hyper:["μξ, ρ²ξ",""]};
  const graphs = {
    "G:original": {nodes:{a:[36,76],tau:[126,76],w:[289,76]},edges:[["a","tau"],["tau","w"]]},
    "G:expanded": {nodes:{a:[25,76],at:[90,76],t:[158,76],w:[289,76]},edges:[["a","at"],["at","t"],["t","w"]]},
    "GL:original": {nodes:{a:[26,76],tau:[127,76],lambda:[221,76],w:[304,76]},edges:[["a","tau"],["tau","lambda"],["lambda","w"]]},
    "GL:expanded": {nodes:{a:[20,89],at:[79,89],t:[144,89],ar:[211,40],tr:[300,40],w:[300,89]},edges:[["a","at"],["at","t"],["ar","tr"],["tr","w"],["t","w"]]},
    "GLT:original": {nodes:{hyper:[34,36],xi:[129,36],K:[35,91],tau:[129,91],lambda:[222,91],w:[304,91]},edges:[["hyper","xi"],["xi","tau"],["K","tau"],["xi","lambda"],["tau","lambda"],["lambda","w"]]},
    "GLT:expanded": {nodes:{hyper:[31,37],xi:[121,37],K:[30,91],tau:[121,91],nu:[219,37],lambda:[219,91],w:[301,91]},edges:[["hyper","xi"],["xi","tau"],["K","tau"],["xi","nu"],["tau","lambda"],["nu","lambda"],["lambda","w"]]}
  };
  function el(tag,attrs={},txt) {
    const node=document.createElementNS(ns,tag);
    for(const [k,v] of Object.entries(attrs)) node.setAttribute(k,v);
    if(txt!==undefined) node.textContent=txt;
    return node;
  }
  window.drawHierarchy = function(svg,prior,form) {
    const {nodes,edges}=graphs[prior+":"+form];
    svg.setAttribute("viewBox","0 0 345 142");
    svg.setAttribute("role","img");
    svg.setAttribute("aria-labelledby","graph-title graph-description");
    const describe = name=>labels[name].join("_");
    const def=el("defs"),marker=el("marker",{id:"prior-arrow",viewBox:"0 0 8 8",refX:7,refY:4,markerWidth:6,markerHeight:6,orient:"auto-start-reverse"});
    marker.append(el("path",{d:"M0 0 L8 4 L0 8 Z",fill:"#667e83"}));def.append(marker);
    svg.replaceChildren(el("title",{id:"graph-title"},prior+"-BNN "+form+" hierarchy"),el("desc",{id:"graph-description"},
      "Conditional on the observation variance sigma squared. "+edges.map(([from,to])=>describe(from)+" determines the law of "+describe(to)).join(". ")+". Unit-specific nodes repeat in the plate."),def);
    svg.append(el("rect",{x:185,y:12,width:153,height:121,rx:8,fill:"#f8faf8",stroke:"#a9bcb8","stroke-dasharray":"4 3"}));
    const plate=el("text",{x:328,y:125,"text-anchor":"end","font-size":11,fill:"#506969"},"k = 1, …, K");
    plate.append(el("tspan",{"baseline-shift":"sub","font-size":8},"l"));svg.append(plate);
    for(const [from,to] of edges){
      const [x,y]=nodes[from],[u,v]=nodes[to],d=Math.hypot(u-x,v-y),dx=(u-x)/d,dy=(v-y)/d;
      svg.append(el("line",{x1:x+20*dx,y1:y+20*dy,x2:u-20*dx,y2:v-20*dy,stroke:"#667e83","stroke-width":1.45,"marker-end":"url(#prior-arrow)"}));
    }
    for(const [key,[x,y]] of Object.entries(nodes)){
      const weight=key==="w",fixed=key==="a"||key==="K"||key==="hyper";
      if(!fixed)svg.append(el("circle",{cx:x,cy:y,r:18,fill:weight?"#e4f0ee":"#fffefa",stroke:weight?"#135d7a":"#708387","stroke-width":weight?2.8:1.25}));
      const text=el("text",{x,y:y+4,"text-anchor":"middle",fill:"#213b43","font-family":"Cambria Math, Georgia, serif","font-size":17,"font-style":"italic","font-weight":weight?700:400},labels[key][0]);
      if(key==="hyper") {
        text.textContent="μ"; text.append(el("tspan",{"baseline-shift":"sub","font-size":10},"ξ"),
          el("tspan",{},", ρ"), el("tspan",{"baseline-shift":"super","font-size":10},"2"),
          el("tspan",{"baseline-shift":"sub","font-size":10},"ξ"));
      } else text.append(el("tspan",{"baseline-shift":"sub","font-size":10},labels[key][1]));
      svg.append(text);
    }
  };
})();
