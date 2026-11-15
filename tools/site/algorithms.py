"""Render the offline algorithm walkthrough and its local R source reference."""
from html import escape
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from xml.etree import ElementTree

from latex2mathml.converter import convert

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = ROOT / "docs/computation"
PRIORS = ("G", "GL", "GLT")
R_DEFINITION = re.compile(r"^([.A-Za-z][\w.]*)\s*<-\s*", re.M)
BLOCK_TEMPLATE = HERE / "algorithms.html.in"


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def equation(tex, block=False):
    # Preserve two-column alignment using a MathML table supported by the converter.
    source = tex.replace(r"\begin{aligned}", r"\begin{array}{rl}").replace(r"\end{aligned}", r"\end{array}")
    math = convert(source, display="block" if block else "inline")
    ElementTree.fromstring(math)
    math = math.replace("<math ", f'<math alttext="{escape(tex, quote=True)}" ', 1)
    return f'<div class="equation" tabindex="0">{math}</div>' if block else math


def prose(value):
    parts = []
    position = 0
    for token in re.finditer(r"`([^`\n]+)`|(?<!\\)\$([^$\n]+)\$", value):
        parts.append(escape(value[position:token.start()]))
        parts.append(f"<code>{escape(token[1])}</code>" if token[1] is not None else equation(token[2]))
        position = token.end()
    return "".join(parts) + escape(value[position:])


def blocks(items):
    html = []
    for item in items:
        if set(item) == {"text"}:
            html.append(f'<p>{prose(item["text"])}</p>')
        elif set(item) == {"math"}:
            html.append(equation(item["math"], True))
        elif set(item) == {"code"}:
            html.append(f'<pre tabindex="0"><code>{escape(item["code"])}</code></pre>')
        elif set(item) == {"detail", "blocks"}:
            html.append(f'<details class="notes"><summary>{prose(item["detail"])}</summary>{blocks(item["blocks"])}</details>')
        else:
            raise ValueError(f"Unknown step block keys {set(item)}")
    return "\n".join(html)


def local_source():
    entries, anchors = [], set()
    for path in sorted((ROOT/"R").glob("*.R")):
        code = path.read_text(encoding="utf8")
        matches = list(R_DEFINITION.finditer(code))
        for i, match in enumerate(matches):
            key = slug(match[1])
            if key in anchors:
                raise ValueError(f"Duplicate R definition anchor: {key}")
            anchors.add(key)
            end = matches[i+1].start() if i+1 < len(matches) else len(code)
            # Code before the following function may include a descriptive comment for it.
            snippet = code[match.start():end].rstrip()
            while snippet.splitlines()[-1].lstrip().startswith("#"):
                snippet = snippet.rsplit("\n",1)[0].rstrip()
            line = code[:match.start()].count("\n") + 1
            entries.append((key, match[1], path.name, line, snippet))
    toc = " ".join(f'<a href="#{key}">{escape(name)}</a>' for key,name,*_ in entries)
    body = "\n".join(f'<section id="{key}"><p class="location">R/{file} · line {line}</p><h2><code>{escape(name)}</code></h2><pre tabindex="0"><code>{escape(snippet)}</code></pre></section>' for key,name,file,line,snippet in entries)
    common_css = (ROOT/"docs/package.css").read_text(encoding="utf8")
    source = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R implementation · gltbnn</title><style>{common_css}
main {{max-width:62rem;margin:auto;padding:1rem clamp(1rem,4vw,3rem) 3rem}}h1 {{font-size:clamp(2.4rem,6vw,3.8rem);margin-block:1.25rem}}h2 {{font-size:1.35rem;overflow-wrap:anywhere}}section {{border-top:1px solid var(--rule);margin-top:2rem}}pre {{background:#f1f1eb;border-radius:4px}}.location,summary {{color:var(--muted)}}details a {{display:inline-block;margin:.18rem .4rem;overflow-wrap:anywhere}}summary {{cursor:pointer}}</style></head><body><main><a href="algorithms.html">Algorithm walkthrough</a><h1>R implementation</h1><p>These functions and quadrature constants come from the package source. File names and starting line numbers identify each definition in this release.</p><details><summary>Definition index</summary>{toc}</details>{body}</main></body></html>'''
    (DEST/"source.html").write_bytes(source.encode())
    return anchors


def model_html(prior, model, anchors):
    listed, panels, ids = [], [], set()
    for i, step in enumerate(model["steps"],1):
        key = step["id"]
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*",key) or key in ids:
            raise ValueError(f"Invalid or duplicate step ID: {prior}/{key}")
        ids.add(key)
        anchor = prior.lower() + "-" + key
        listed.append(f'<li><a class="step-link" href="#{anchor}" data-key="{key}">{escape(step["title"])}</a></li>')
        summary = f'<p class="lead">{prose(step["summary"])}</p>' if step.get("summary") else ""
        refs = []
        for src in step.get("sources",[]):
            link = src.get("href")
            label = prose(src["label"])
            if link:
                parsed = urlsplit(link)
                if parsed.scheme or parsed.netloc or parsed.path.startswith(("/","\\")) or "\\" in link:
                    raise ValueError(f"Source link must be relative: {link}")
                if parsed.path == "source.html" and parsed.fragment not in anchors:
                    raise ValueError(f"R definition missing: {link}")
                refs.append(f'<li><a href="{escape(link,quote=True)}">{label}</a></li>')
            else:
                refs.append(f'<li>{label}</li>')
        citations = '<details class="notes refs"><summary>Implementation and references</summary><ul>'+''.join(refs)+'</ul></details>' if refs else ''
        panels.append(f'<article class="step" id="{anchor}" data-key="{key}"><p class="eyebrow">{prior}-BNN · Step {i} of {len(model["steps"])}</p><h3 tabindex="-1">{escape(step["title"])}</h3>{summary}{blocks(step["blocks"])}{citations}</article>')
    if not listed:
        raise ValueError(f"No steps for {prior}")
    return f'''<section class="model" data-prior="{prior}" id="{prior.lower()}"><div class="model-head"><h2>{escape(model['title'])}</h2><p>{prose(model['summary'])}</p></div><div class="layout"><details class="step-index" open><summary>Steps in {prior}-BNN</summary><ol>{''.join(listed)}</ol></details><div class="reading">{''.join(panels)}<div class="paging" hidden><button type="button" data-move="-1">← Previous step</button><a class="permalink" href="#{prior.lower()}">Step link</a><button type="button" data-move="1">Next step →</button></div></div></div></section>'''


def render():
    data = json.loads((HERE/"algorithm.json").read_text(encoding="utf8"))
    if set(data["priors"]) != set(PRIORS):
        raise ValueError("Expected G, GL and GLT.")
    for model in data["priors"].values():
        model["steps"] = [data.get("shared_steps", {})[s] if isinstance(s, str) else s for s in model["steps"]]
    anchors = local_source()
    inserts = {"TITLE":escape(data["title"]),"INTRO":prose(data["intro"]),"MODELS":"\n".join(model_html(p,data["priors"][p],anchors) for p in PRIORS),"CSS":(ROOT/"docs/package.css").read_text(encoding="utf8")}
    output = BLOCK_TEMPLATE.read_text(encoding="utf8")
    for key, value in inserts.items():
        marker = "@@"+key+"@@"
        if marker not in output: raise ValueError(f"Missing {marker}")
        output = output.replace(marker,value)
    (DEST/"algorithms.html").write_bytes(output.encode())


if __name__ == "__main__":
    render()
