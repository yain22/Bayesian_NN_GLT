"""Render the explorer's TeX as native MathML. Run with Python and latex2mathml.

The deployed pages and the R package do not require Python.
"""
from pathlib import Path
import re
import unicodedata
from html import escape
from latex2mathml.converter import convert

source = Path(__file__).resolve().parent
src = (source / 'explorer.html.in').read_text(encoding='utf8')
topics = {'spread':'Weight prior','scales':'Layer &amp; coefficient scales','hierarchy':'Prior hierarchies','tail':'Tail probability','prune':'Pruning'}
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
    'G': r'\[\lambda_{[[kl]],[[j]]}=\tau_{[[l]]}.\] All [[m]] coefficients in layer [[l]] share the scalar \(\tau_{[[l]]}\).',
    'GL': r'\[\lambda_{[[kl]],[[j]]}=\tau_{[[l]]} r_{[[kl]],[[j]]},\quad r_{[[kl]],[[j]]}\sim C^+(0,1).\] Coordinate [[j]] has its own half-Cauchy multiplier. At quantile \(u\), \(r_{[[kl]],[[j]]}=\tan(\pi u/2)\).',
    'GLT': r'\[\lambda_{[[kl]],[[j]]}\mid\tau_{[[l]]},\xi_{[[l]]}\sim\operatorname{GPD}(\tau_{[[l]]},\xi_{[[l]]}).\] At quantile \(u\), \(\lambda_{[[kl]],[[j]]}=(\tau_{[[l]]}/\xi_{[[l]]})\{(1-u)^{-\xi_{[[l]]}}-1\}\). The Tail probability tab sets \(\xi_{[[l]]}\) for layer [[l]].'
}

def choice(prior, body, form=None):
    attr = f'data-law="{prior}"' if form is None else f'data-hierarchy="{prior}:{form}"'
    return f'<div {attr} hidden>{body}</div>'

src = src.replace('@scalelaws@',''.join(choice(p,s) for p,s in scales.items()))
layer = r'\[\tau_{[[l]]}\sim C^+(0,a_{[[l]]}),\quad a_{[[l]]}=1/M_{[[l]]}=1/[[m]].\]'
base = r'\[t_{[[l]]}=\tau_{[[l]]}^2,\quad a_{[[l]]}=1/[[m]],\]\[a_{\tau,[[l]]}\sim\operatorname{IG}(1/2,a_{[[l]]}^{-2}),\]\[t_{[[l]]}\mid a_{\tau,[[l]]}\sim\operatorname{IG}(1/2,1/a_{\tau,[[l]]}).\]'
tail = r'\[\log\xi_{[[l]]}\sim\mathcal N(0,1)\big|_{(-\log 2,\infty)},\]\[\tau_{[[l]]}\mid\xi_{[[l]]}\sim\operatorname{IG}(1+[[m]]/\xi_{[[l]]},1).\]'
laws = {
    ('G','original'): layer + r'\[\lambda_{[[kl]],[[j]]}=\tau_{[[l]]}.\] All [[m]] hidden coefficients in this layer share this scale.',
    ('G','expanded'): base + r'Integrating over \(a_{\tau,[[l]]}\) recovers the half-Cauchy law for \(\sqrt{t_{[[l]]}}\).',
    ('GL','original'): layer + r'\[r_{[[kl]],[[j]]}\sim C^+(0,1),\quad\lambda_{[[kl]],[[j]]}=\tau_{[[l]]}r_{[[kl]],[[j]]}.\] Each coefficient draws an independent multiplier, independently of the layer scale.',
    ('GL','expanded'): base + r'\[t_{r,[[kl]],[[j]]}=r_{[[kl]],[[j]]}^2,\quad a_{r,[[kl]],[[j]]}\sim\operatorname{IG}(1/2,1),\]\[t_{r,[[kl]],[[j]]}\mid a_{r,[[kl]],[[j]]}\sim\operatorname{IG}(1/2,1/a_{r,[[kl]],[[j]]}).\] Integrating out the auxiliary variables recovers the half-Cauchy scales.',
    ('GLT','original'): tail + r'\[\lambda_{[[kl]],[[j]]}\mid\tau_{[[l]]},\xi_{[[l]]}\sim\operatorname{GPD}(\tau_{[[l]]},\xi_{[[l]]}).\] Given the layer parameters, its [[m]] scalar local scales are independent.',
    ('GLT','expanded'): tail + r'\[\nu_{[[kl]],[[j]]}\mid\xi_{[[l]]}\sim\operatorname{IG}(1/\xi_{[[l]]},1/\xi_{[[l]]}),\]\[\lambda_{[[kl]],[[j]]}\mid\tau_{[[l]]},\nu_{[[kl]],[[j]]}\sim\operatorname{Exp}(\tau_{[[l]]}\nu_{[[kl]],[[j]]}).\] The exponential has mean \(\tau_{[[l]]}\nu_{[[kl]],[[j]]}\). Integrating out the inverse-gamma mixing variable gives the generalized Pareto law.'
}
src = src.replace('@hierarchies@','\n'.join(choice(p,v,f) for (p,f),v in laws.items()))

# Represent the diagonal compactly; each scalar variance is expanded below.
global_var = r'\sigma^2\tau_{[[l]]}^2'
squared_var = r'\sigma^2t_{[[l]]}'
local_var = r'\sigma^2\lambda_{[[kl]],COORD}^2'
product_var = r'\sigma^2t_{[[l]]}t_{r,[[kl]],COORD}'
variances = {'global': global_var, 'squared-layer': squared_var,
             'local': local_var, 'squared-product': product_var}

def variants(expressions):
    return ''.join(f'<span data-weight-law="{kind}" hidden>\\({tex}\\)</span>'
                   for kind,tex in expressions.items())

vector_head = r'\mathbf w_{[[kl]]}\sim\mathcal N(\mathbf0,'
covariances = {
    'global': global_var+r'\mathbf I_{[[d]]}',
    'squared-layer': squared_var+r'\mathbf I_{[[d]]}',
    'local': r'\sigma^2\operatorname{diag}(\lambda_{[[kl]],1}^2,\ldots,\lambda_{[[kl]],[[d]]}^2)',
    'squared-product': r'\sigma^2t_{[[l]]}\operatorname{diag}(t_{r,[[kl]],1},\ldots,t_{r,[[kl]],[[d]]})'
}
src = src.replace('@selected-vector-laws@', variants({n:vector_head+v+')' for n,v in covariances.items()}))
src = src.replace('@selected-variances@', variants({n:v.replace('COORD','[[j]]') for n,v in variances.items()}))
leaves = []
for j in [1, 2, 3, 4, None]:
    if j is None:
        coordinate, variable, origin = '[[d]]', r'b_{[[kl]]}', r'Bias input \(1\)'
    else:
        coordinate, variable = str(j), rf'w_{{[[kl]],{j}}}'
        input_source = rf'<span data-source-layer="1">Input \(x_{{{j}}}\)</span>' if j <= 3 else ''
        hidden_source = rf'<span data-source-layer="2">Activation \(h_{{{j}1}}(\mathbf x)\)</span>'
        origin = input_source + hidden_source
    normal = variants({n:variable+r'\sim\mathcal N(0,'+v.replace('COORD',coordinate)+')' for n,v in variances.items()})
    layer_attr = ' data-source-layer="2"' if j == 4 else ''
    leaves.append(f'<li class="coordinate-leaf" data-coordinate="{j or "bias"}"{layer_attr}><span class="coordinate-source">{origin}</span>'
                  f'<span class="coordinate-law">{normal}</span></li>')
src = src.replace('@selected-leaves@', ''.join(leaves))
initial = {'k':'2', 'l':'1', 'kl':'21', 'prev':'0', 'width':'4', 'd':'4', 'inputs':'3', 'j':'1', 'm':'16'}
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
    # Use the bold Greek glyph in browsers implementing MathML Core.
    for name, letter in [('BETA', 'β'), ('LAMDA', 'λ')]:
        bold = unicodedata.lookup('MATHEMATICAL BOLD ITALIC SMALL ' + name)
        m = m.replace(f'<mi mathvariant="bold-italic">&#x{ord(letter):05X};</mi>',
                      f'<mi>&#x{ord(bold):X};</mi>')
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
