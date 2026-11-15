"""Build a self-contained replay from the recorded, checksum-verified fitting traces."""
import csv
import io
import json
import math
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile


SETTINGS = ("Linear", "Reference additive", "Interaction", "Smaller sample",
            "Larger sample", "500 predictors", "30 active inputs", "Higher correlation",
            "Lower signal-to-noise ratio")
PRIORS = ("G", "GL", "GLT")


def recorded_traces(archive):
    key = "multistart/traces.csv"
    with ZipFile(archive) as z:
        records = z.read(key)
        manifest = dict(line.split("  ")[::-1] for line in z.read("SHA256SUMS").decode().splitlines())
    if sha256(records).hexdigest() != manifest[key]:
        raise ValueError("Trace checksum mismatch")
    traces = {f"C{i}": {p: [[None] * 61 for _ in range(10)] for p in PRIORS} for i in range(1, 10)}
    for row in csv.DictReader(io.StringIO(records.decode("utf8"))):
        start, cycle, value = int(row["start"]), int(row["cycle"]), float(row["elbo"])
        if not (1 <= start <= 10 and 0 <= cycle <= 60 and math.isfinite(value) and value < 0):
            raise ValueError("Unexpected start, fitting cycle or objective value")
        curve = traces[row["setting"]][row["prior"]][start-1]
        if curve[cycle] is not None:
            raise ValueError("Duplicate trace observation")
        curve[cycle] = value
    if any(None in curve for setting in traces.values() for starts in setting.values() for curve in starts):
        raise ValueError("Incomplete monitor trace")
    return traces


def build():
    tools_dir = Path(__file__).resolve().parent
    output_dir = tools_dir.parents[1]/"docs/computation"
    data = recorded_traces(output_dir/"elbo-initializations.zip")
    page = (tools_dir/"elbo_replay.html.in").read_text("utf8")
    selected = data["C2"]
    log_values = [math.log10(-v) for starts in selected.values() for curve in starts for v in curve]
    lo, hi = math.floor(min(log_values)), math.ceil(max(log_values))
    paths, ticks, vals = [], [], []
    for p in PRIORS:
        curve = selected[p][0]
        coords = " ".join(f"{i*100/60:.6f},{100*(math.log10(-v)-lo)/(hi-lo):.6f}" for i,v in enumerate(curve))
        paths.append(f'<polyline id="line-{p}" class="trace {p}" points="{coords}"/>')
        vals.append(f'<tr><th scope="row"><span class="key {p}"></span>{p}</th><td id="value-{p}">{curve[-1]:,.1f}</td></tr>')
    for power in range(lo, hi+1):
        ticks.append(f'<span class="tick" style="top:{100*(power-lo)/(hi-lo):.4f}%">−10<sup>{power}</sup></span>')
    options = [f'<option value="C{i}" {"selected" if i==2 else ""}>C{i} · {s}</option>' for i,s in enumerate(SETTINGS,1)]
    replacements = {"@@DATA@@":json.dumps(data,separators=(",",":"),allow_nan=False),
                    "@@PATHS@@":"".join(paths), "@@TICKS@@":"".join(ticks),
                    "@@VALUES@@":"".join(vals), "@@SETTINGS@@":"".join(options),
                    "@@STARTS@@":"".join(f"<option>{i}</option>" for i in range(1,11))}
    for marker, value in replacements.items():
        if page.count(marker) != 1:
            raise ValueError("Missing or repeated replay template marker")
        page = page.replace(marker,value)
    (output_dir/"elbo-replay.html").write_bytes(page.encode("utf8"))


if __name__ == "__main__":
    build()
