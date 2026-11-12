"""Build GitHub Pages assets from the verified manuscript reports."""
from argparse import ArgumentParser
import csv
from hashlib import sha256
from itertools import combinations, product
import json
import math
from pathlib import Path
from shutil import copyfile
import subprocess
import tempfile


def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_intervals(records, coverage, width, label):
    for row in records:
        values=[row[coverage],row[width]]
        if row["method"] in ("fnn","dropout","l1") and all(not v or v=="NA" for v in values):
            continue
        probability,span=map(float,values)
        require(math.isfinite(probability) and 0<=probability<=1 and math.isfinite(span) and span>=0,
                f"Invalid {coverage} or {width} in the {label} table.")


def build(root,page):
    main=root/"main"
    inventory=rows(main/"v3_main_report_hashes.csv")
    expected_files={"v3_main_provenance.txt","v3_design_rmse.csv","v3_paired_differences.csv",
                    "v3_design_scales.csv","v3_interval_accuracy.csv"}
    members=[r["file"] for r in inventory]
    require(len(members)==len(set(members)) and expected_files.issubset(members),
            "The main report inventory is incomplete or contains duplicates.")
    for row in inventory:
        require(Path(row["file"]).name==row["file"],"Expected a report filename in the inventory.")
        require(sha256((main/row["file"]).read_bytes()).hexdigest()==row["sha256"],
                f"Checksum mismatch in {row['file']}.")
    require("Verified fit records 57600" in (main/"v3_main_provenance.txt").read_text().splitlines(),
            "The completed main fit count is missing.")
    check = """
      base <- commandArgs(trailingOnly=TRUE)[1]
      receipt <- readRDS(file.path(base,"application_reports_receipt.rds"))
      required <- c("engineering/reports/across_splits.csv", "sensitivity/reports/by_setting.csv",
                    "year/reports/records.csv", "housing/reports/records.csv")
      stopifnot(identical(receipt$version,"v3-coefficient"),
        identical(receipt$counts,c(engineering=1188L,sensitivity=1280L,year=36L,housing=36L)),
        identical(receipt$verified,"recomputed against the archived test responses"),
        !anyDuplicated(receipt$files), all(required %in% receipt$files),
        all(grepl("^(engineering|sensitivity|year|housing)/reports/[a-z_]+[.]csv$",receipt$files)),
        identical(unname(vapply(file.path(base,receipt$files),function(f)
          digest::digest(file=f,algo="sha256"),"")),receipt$sha256))
    """
    with tempfile.TemporaryDirectory() as tmp:
        script=Path(tmp)/"verify.R"
        script.write_text(check,encoding="utf-8")
        subprocess.run(["Rscript","--vanilla",str(script),str(root)],check=True)
    methods=("g","hs","glt","fnn","dropout","l1")
    keys={"/".join(map(str,indices)) for indices in product(range(1,9),(1,3),(2,5,10),(10,25))}
    key=lambda r:"/".join(r[f] for f in ("cell","snr","depth","width"))
    content={k:{"loss":{},"scales":{}} for k in sorted(keys)}
    contrasts={}
    pair_rows=rows(main/"v3_paired_differences.csv")
    all_pairs=set(product(keys,(frozenset(pair) for pair in combinations(methods,2))))
    seen=set()
    for r in pair_rows:
        combination=(key(r),frozenset((r["first"],r["second"])))
        require(combination in all_pairs and combination not in seen,
                "Missing or duplicate method pair within a simulation setting.")
        seen.add(combination)
        for target in ("rmse","mean_rmse"):
            estimate=float(r[target])
            mcse=float(r[target+"_mcse"])
            require(math.isfinite(estimate) and math.isfinite(mcse) and mcse>=0,
                    "Invalid paired loss estimate or MCSE.")
        if "glt" not in (r["first"],r["second"]):
            continue
        other,sign=(r["second"],-1) if r["first"]=="glt" else (r["first"],1)
        for target in ("rmse","mean_rmse"):
            identity=(key(r),other,target)
            contrasts[identity]=[sign*float(r[target]),float(r[target+"_mcse"])]
    require(seen==all_pairs,"The study requires all 15 pairs in each of its 96 settings.")
    losses=rows(main/"v3_design_rmse.csv")
    require(len(losses)==96*6,"Incomplete loss table.")
    for r in losses:
        k,m=key(r),r["method"]
        require(m not in content[k]["loss"],"Repeated method in the loss table.")
        content[k]["loss"][m]={t:[float(r[t]),*(contrasts[k,m,t] if m!="glt" else [None,None])]
                               for t in ("rmse","mean_rmse")}
    scales=rows(main/"v3_design_scales.csv")
    require(len(scales)==96*3,"Incomplete table of first-layer scales.")
    for r in scales:
        k,m=key(r),r["method"]
        require(m not in content[k]["scales"],"Repeated prior in the table of first-layer scales.")
        content[k]["scales"][m]={t:[float(r[t]),float(r[t+"_mcse"])] for t in
                                ("active_minus_inactive","within_unit_log_ratio")}
    for row in content.values():
        require(set(row["loss"])==set(methods) and set(row["scales"])==set(methods[:3]),
                "The reported methods do not match the simulation design.")
        for m in methods:
            for t,(loss,delta,mcse) in row["loss"][m].items():
                require(math.isfinite(loss) and loss>=0,"Invalid RMSE in the loss table.")
        for obj in row["scales"].values():
            require(all(math.isfinite(mu) and math.isfinite(se) and se>=0 for mu,se in obj.values()),
                    "Nonfinite scale estimate or invalid MCSE.")
        require(all(pair==[0.,0.] for pair in row["scales"]["g"].values()),
                "A common layer scale must give zero within-layer contrasts under G.")
    for r in pair_rows:
        loss=content[key(r)]["loss"]
        for target in ("rmse","mean_rmse"):
            delta=loss[r["first"]][target][0]-loss[r["second"]][target][0]
            require(math.isclose(delta,float(r[target]),abs_tol=1e-11),
                    "A reported contrast disagrees with its two method means.")
    # Only aggregate statistics are distributed. The housing outcome unit is
    # $100,000 as in the benchmark. Song-year errors are in calendar years.
    public_tables={name:main/name for name in ("v3_design_rmse.csv","v3_paired_differences.csv",
                   "v3_design_scales.csv","v3_interval_accuracy.csv")}
    interval_rows=rows(public_tables["v3_interval_accuracy.csv"])
    check_intervals(interval_rows,"mean_coverage","mean_width","C2")
    check_intervals(interval_rows,"response_coverage","response_width","C2")
    apps={"engineering":"across_splits.csv","sensitivity":"by_setting.csv",
          "year":"records.csv","housing":"records.csv"}
    selected={}
    for name,filename in apps.items():
        data=rows(root/name/"reports"/filename)
        expected={"engineering":108,"sensitivity":128,"year":36,"housing":36}[name]
        require(len(data)==expected,f"Incomplete {name} table.")
        if name=="sensitivity":
            for target in ("mean","response"):
                check_intervals(data,target+"_coverage90",target+"_width90",name)
        else:
            check_intervals(data,"coverage90","width90",name)
        if name in ("year","housing"):
            grid={(int(d["depth"]),int(d["width"]),d["method"]) for d in data}
            require(grid==set(product((2,5,10),(10,25),methods)),f"Incomplete {name} design.")
            sizes=(463715,51630) if name=="year" else (13209,4128)
            require(all((int(d["n_train"]),int(d["n_test"]))==sizes and
                        all(math.isfinite(float(d[k])) and float(d[k])>=0 for k in ("rmse","mae"))
                        for d in data), f"Invalid sample size or prediction error in {name}.")
            selected[name]=[{field:d[field] for field in ("depth","width","method","n_train","n_test",
                "rmse","mae","coverage90","width90")} for d in data]
        else:
            public_tables[name+".csv"]=root/name/"reports"/filename
    out=page/"data"
    out.mkdir(exist_ok=True,parents=True)
    for name,path in public_tables.items():
        copyfile(path,out/name)
    for name,data in selected.items():
        with (out/(name+".csv")).open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (page/"measurements.js").write_text("const studyMeasurements = "+json.dumps(content,separators=(",",":"),allow_nan=False)+";\n",encoding="utf-8")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("reports",type=Path,help="Directory containing the independently validated study reports")
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[2]/"docs"/"study")
    args=p.parse_args()
    build(args.reports,args.output)
