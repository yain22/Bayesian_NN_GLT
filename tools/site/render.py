"""Render the explorer's TeX as native MathML. Run with Python and latex2mathml.

The deployed pages and the R package do not require Python.
"""
from pathlib import Path
import re
from html import escape
from latex2mathml.converter import convert

source = Path(__file__).resolve().parent
src = (source / 'explorer.html.in').read_text(encoding='utf8')
topics = {'spread':'Weight prior','scales':'Layer &amp; unit scales','hierarchy':'Prior hierarchies','tail':'Tail probability','prune':'Pruning'}
tabs = []
for i,(key,title) in enumerate(topics.items()):
    active = 'true' if i == 0 else 'false'
    focus = 0 if i == 0 else -1
    hidden = '' if i == 0 else ' hidden'
    tabs.append(f'<button class="tab" type="button" id="tab-{key}" role="tab" aria-selected="{active}" aria-controls="panel-{key}" tabindex="{focus}">{title}</button>')
    src = src.replace(f'@panel {key}@', f'<section class="pane" id="panel-{key}" role="tabpanel" aria-labelledby="tab-{key}" tabindex="0"{hidden}>')
src = src.replace('@tabs@','\n'.join(tabs))
buttons = [f'<button class="choice" type="button" data-prior="{p}" aria-pressed="{str(p=="GL").lower()}">{p}-BNN</button>' for p in ('G','GL','GLT')]
src = src.replace('@priors@','<div class="choice-group" role="group" aria-label="Prior family">'+''.join(buttons)+'</div>')
scales = {
'G': r'\[\lambda_{[[kl]]}=\tau_{[[l]]}.\] All [[width]] units in layer [[l]], including unit [[k]], use the same scale \(\tau_{[[l]]}\).',
'GL': r'\[\lambda_{[[kl]]}=\tau_{[[l]]} r_{[[kl]]},\quad r_{[[kl]]}\sim C^+(0,1).\] Unit [[k]] has its own half-Cauchy multiplier \(r_{[[kl]]}\). Its value at quantile \(u\) is \(r_{[[kl]]}=\tan(\pi u/2)\).',
'GLT': r'\[\lambda_{[[kl]]}\mid\tau_{[[l]]},\xi_{[[l]]}\sim\operatorname{GPD}(\tau_{[[l]]},\xi_{[[l]]}).\] At quantile \(u\), \(\lambda_{[[kl]]}=(\tau_{[[l]]}/\xi_{[[l]]})\{(1-u)^{-\xi_{[[l]]}}-1\}\). The Tail probability tab sets \(\xi_{[[l]]}\) for layer [[l]].'}
def choice(prior, body, form=None):
    attr = f'data-law="{prior}"' if form is None else f'data-hierarchy="{prior}:{form}"'
    return f'<div {attr} hidden>{body}</div>'
src = src.replace('@scalelaws@',''.join(choice(p,s) for p,s in scales.items()))
layer = r'\[\tau_{[[l]]}\sim C^+(0,a_{[[l]]}),\quad a_{[[l]]}=1/[[width]].\]'
base = r'\[t_{[[l]]}=\tau_{[[l]]}^2,\quad a_{[[l]]}=1/[[width]],\]\[a_{\tau,[[l]]}\sim\operatorname{IG}(1/2,a_{[[l]]}^{-2}),\]\[t_{[[l]]}\mid a_{\tau,[[l]]}\sim\operatorname{IG}(1/2,1/a_{\tau,[[l]]}).\]'
tail = r'\[\log\xi_{[[l]]}\sim\mathcal N(0,1)\big|_{(-\log 2,\infty)},\]\[\tau_{[[l]]}\mid\xi_{[[l]]}\sim\operatorname{IG}(1+[[width]]/\xi_{[[l]]},1).\]'
laws = {
('G','original'): layer + r'\[\lambda_{[[kl]]}=\tau_{[[l]]}.\] All [[width]] incoming weight vectors in layer [[l]] share this scale.',
('G','expanded'): base + r'<p>The conditional covariance of \(\mathbf w_{[[kl]]}\) is \(\sigma^2t_{[[l]]}\mathbf I_{[[d]]}\). Integrating over \(a_{\tau,[[l]]}\) recovers the half-Cauchy law for \(\sqrt{t_{[[l]]}}\).</p>',
('GL','original'): layer + r'\[r_{[[kl]]}\sim C^+(0,1),\quad\lambda_{[[kl]]}=\tau_{[[l]]}r_{[[kl]]}.\] Each unit draws its multiplier independently of the other multipliers and of \(\tau_{[[l]]}\). For this unit, \(\lambda_{[[kl]]}\mid\tau_{[[l]]}\sim C^+(0,\tau_{[[l]]})\).',
('GL','expanded'): base + r'\[t_{r,[[kl]]}=r_{[[kl]]}^2,\quad a_{r,[[kl]]}\sim\operatorname{IG}(1/2,1),\]\[t_{r,[[kl]]}\mid a_{r,[[kl]]}\sim\operatorname{IG}(1/2,1/a_{r,[[kl]]}).\]<p>Integrating over each auxiliary variable recovers its half-Cauchy scale. The conditional covariance of \(\mathbf w_{[[kl]]}\) is \(\sigma^2t_{[[l]]}t_{r,[[kl]]}\mathbf I_{[[d]]}\).</p>',
('GLT','original'): tail + r'\[\lambda_{[[kl]]}\mid\tau_{[[l]]},\xi_{[[l]]}\sim\operatorname{GPD}(\tau_{[[l]]},\xi_{[[l]]}).\] Given the layer parameters, its [[width]] unit scales are independent. The displayed hyperparameters are \((\mu_\xi,\rho_\xi^2)=(0,1)\), with \(\xi_{[[l]]}>1/2\).',
('GLT','expanded'): tail + r'\[\nu_{[[kl]]}\mid\xi_{[[l]]}\sim\operatorname{IG}(1/\xi_{[[l]]},1/\xi_{[[l]]}),\]\[\lambda_{[[kl]]}\mid\tau_{[[l]]},\nu_{[[kl]]}\sim\operatorname{Exp}(\tau_{[[l]]}\nu_{[[kl]]}).\] The exponential has mean \(\tau_{[[l]]}\nu_{[[kl]]}\). Integrating over \(\nu_{[[kl]]}\) recovers the generalized Pareto law.'}
src = src.replace('@hierarchies@','\n'.join(choice(p,v,f) for (p,f),v in laws.items()))

# The vector and its coordinates use the same variance expression in every hierarchy.
weight_laws = {
    'global': (r'\tau_{[[l]]}', r'\sigma^2\tau_{[[l]]}^2'),
    'squared-layer': (r't_{[[l]]}', r'\sigma^2t_{[[l]]}'),
    'local': (r'\lambda_{[[kl]]}', r'\sigma^2\lambda_{[[kl]]}^2'),
    'squared-product': (r't_{[[l]]},t_{r,[[kl]]}', r'\sigma^2t_{[[l]]}t_{r,[[kl]]}'),
}

def weight_variants(equation):
    return ''.join(
        f'<span data-weight-law="{kind}" hidden>\\({equation(scales,variance)}\\)</span>'
        for kind,(scales,variance) in weight_laws.items()
    )

vector = weight_variants(lambda scales, variance:
    r'\mathbf w_{[[kl]]}\mid '+scales+r',\sigma^2\sim\mathcal N(\mathbf0,'+variance+r'\mathbf I_{[[d]]})')
src = src.replace('@selected-vector-laws@', vector)
src = src.replace('@selected-conditions@', weight_variants(lambda scales, variance: scales+r',\sigma^2'))
src = src.replace('@selected-variances@', weight_variants(lambda scales, variance: variance))
leaves = []
for j in [1, 2, 3, 4, None]:
    if j is None:
        variable, origin = r'b_{[[kl]]}', r'Bias input \(1\)'
    else:
        variable = rf'w_{{[[kl]],{j}}}'
        input_source = rf'<span data-source-layer="1">Input \(x_{{{j}}}\)</span>' if j <= 3 else ''
        hidden_source = rf'<span data-source-layer="2">Activation \(h_{{{j}1}}(\mathbf x)\)</span>'
        origin = input_source + hidden_source
    normal = weight_variants(lambda scales, variance: variable+r'\sim\mathcal N(0,'+variance+')')
    layer_attr = ' data-source-layer="2"' if j == 4 else ''
    leaves.append(f'<li class="coordinate-leaf"{layer_attr}><span class="coordinate-source">{origin}</span>'
                  f'<span class="coordinate-law">{normal}</span></li>')
src = src.replace('@selected-leaves@', ''.join(leaves))
initial = {'k':'2', 'l':'1', 'kl':'21', 'prev':'0', 'width':'4', 'd':'4', 'inputs':'3'}
slot = re.compile(r'\[\[(\w+)\]\]')

def number_element(tag, key):
    return f'<{tag} data-selection="{key}">{initial[key]}</{tag}>'

def render(match):
    if match.group(3):
        return number_element('span', match.group(3))
    inline = match.group(1) == '('
    tex = match.group(2)
    # Convert the formula once. Only its numeric leaves and accessible TeX
    # need to change when the reader selects another unit.
    marked = slot.sub(lambda s: r'\text{selected'+s[1]+'}', tex)
    m = convert(marked,display='inline' if inline else 'block')
    m = re.sub(r'<mtext>selected(\w+)</mtext>',lambda s: number_element('mn',s[1]),m)
    alt = slot.sub(lambda s: initial[s[1]],tex)
    attrs = f'alttext="{escape(alt,quote=True)}"'
    if alt != tex:
        attrs += f' data-selection-tex="{escape(tex,quote=True)}"'
    m = m.replace('<math ', '<math '+attrs+' ', 1)
    return m if inline else '<div class="equation">'+m+'</div>'
src = re.sub(r'\\mathbf\s*([A-Za-z0-9])(?![A-Za-z])',r'\\mathbf{\1}',src)
src = re.sub(r'\\mathcal\s+([A-Za-z])',r'\\mathcal{\1}',src)
src = re.sub(r'\\([\(\[])(.*?)\\[\)\]]|\[\[(\w+)\]\]',render,src,flags=re.S)
assert '@panel' not in src and r'\(' not in src
(source.parents[1]/'docs/illustration/index.html').write_bytes(src.encode('utf8'))
