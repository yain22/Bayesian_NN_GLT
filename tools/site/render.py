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
'G': r'\[\lambda_{kl}=\tau_l,\qquad k=1,\ldots,K_l.\] Every unit has the same conditional weight variance.',
'GL': r'\[\lambda_{kl}=\tau_l r_{kl},\quad r_{kl}\sim C^+(0,1).\] Independent half-Cauchy multipliers give unit-specific shrinkage. At quantile \(u\), \(r_{kl}=\tan(\pi u/2)\).',
'GLT': r'\[\lambda_{kl}\mid\tau_l,\xi_l\sim\operatorname{GPD}(\tau_l,\xi_l).\] At quantile \(u\), \(\lambda_{kl}=(\tau_l/\xi_l)\{(1-u)^{-\xi_l}-1\}\). The Tail probability tab sets \(\xi_l\) for this layer.'}
def choice(prior, body, form=None):
    attr = f'data-law="{prior}"' if form is None else f'data-hierarchy="{prior}:{form}"'
    return f'<div {attr} hidden>{body}</div>'
src = src.replace('@scalelaws@',''.join(choice(p,s) for p,s in scales.items()))
layer = r'\[\tau_l\sim C^+(0,a_l),\quad a_l=1/K_l.\]'
base = r'\[t_l=\tau_l^2,\quad a_l=1/K_l,\]\[a_{\tau,l}\sim\operatorname{IG}(1/2,a_l^{-2}),\]\[t_l\mid a_{\tau,l}\sim\operatorname{IG}(1/2,1/a_{\tau,l}).\]'
tail = r'\[\log\xi_l\sim\mathcal N(0,1)\big|_{(-\log 2,\infty)},\]\[\tau_l\mid\xi_l\sim\operatorname{IG}(1+K_l/\xi_l,1).\]'
laws = {
('G','original'): layer + r'\[\lambda_{kl}=\tau_l.\] All \(K_l\) vectors share the layer scale.',
('G','expanded'): base + r'<p>The conditional covariance of \(\mathbf w_{kl}\) is \(\sigma^2t_l\mathbf I_{d_l}\). Marginalizing \(a_{\tau,l}\) recovers the half-Cauchy law for \(\sqrt{t_l}\).</p>',
('GL','original'): layer + r'\[r_{kl}\overset{\mathrm{iid}}{\sim} C^+(0,1),\quad\lambda_{kl}=\tau_l r_{kl}.\] The multipliers are independent of \(\tau_l\). The graph shows the induced law \(\lambda_{kl}\mid\tau_l\sim C^+(0,\tau_l)\).',
('GL','expanded'): base + r'\[t_{r,kl}=r_{kl}^2,\quad a_{r,kl}\sim\operatorname{IG}(1/2,1),\]\[t_{r,kl}\mid a_{r,kl}\sim\operatorname{IG}(1/2,1/a_{r,kl}).\]<p>Marginalizing each auxiliary variable recovers the corresponding half-Cauchy scale. The conditional covariance is \(\sigma^2t_l t_{r,kl}\mathbf I_{d_l}\).</p>',
('GLT','original'): tail + r'\[\lambda_{kl}\mid\tau_l,\xi_l\overset{\mathrm{iid}}{\sim}\operatorname{GPD}(\tau_l,\xi_l).\] The displayed hyperparameters are \((\mu_\xi,\rho_\xi^2)=(0,1)\), so \(\xi_l>1/2\).',
('GLT','expanded'): tail + r'\[\nu_{kl}\mid\xi_l\sim\operatorname{IG}(1/\xi_l,1/\xi_l),\]\[\lambda_{kl}\mid\tau_l,\nu_{kl}\sim\operatorname{Exp}(\tau_l\nu_{kl}).\] The exponential has mean \(\tau_l\nu_{kl}\). Integrating over \(\nu_{kl}\) recovers the generalized Pareto law.'}
src = src.replace('@hierarchies@','\n'.join(choice(p,v,f) for (p,f),v in laws.items()))
def render(match):
    inline = match.group(1) == '('
    tex = match.group(2)
    m = convert(tex,display='inline' if inline else 'block')
    # Some conversion engines leave single-token bold commands unresolved. Use braced TeX.
    m = m.replace('<math ', '<math alttext="'+escape(tex,quote=True)+'" ', 1)
    return m if inline else '<div class="equation">'+m+'</div>'
src = re.sub(r'\\mathbf\s*([A-Za-z0-9])(?![A-Za-z])',r'\\mathbf{\1}',src)
src = re.sub(r'\\mathcal\s+([A-Za-z])',r'\\mathcal{\1}',src)
src = re.sub(r'\\([\(\[])(.*?)\\[\)\]]',render,src,flags=re.S)
assert '@panel' not in src and r'\(' not in src
(source.parents[1]/'docs/illustration/index.html').write_bytes(src.encode('utf8'))
