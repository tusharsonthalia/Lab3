"""Assemble the Lab 3 submission: lab3_<CNetID>.zip, staged next to the project.

1. checkpoints/selected/: copies of the selected compressed exports.
   - Part B: the 12 confirmed operating points (seed-0 pilot checkpoint + compact artifact).
   - Part C: the latest non-superseded artifacts, with an index linking each file to its run_id.
2. Stage the files the handout requires (no datasets, venv, intermediates, or orchestration notes).
3. MANIFEST.sha256 over every staged file.
4. Zip.

Usage: uv run python scripts/build_submission.py [--cnetid sonthalt]
Put memo.pdf in the project root first. The script warns if it is missing.
"""
import argparse, hashlib, json, os, shutil, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--cnetid", default="sonthalt"); args = ap.parse_args()
STAGE = ROOT / "submission" / f"lab3_{args.cnetid}"
ZIP = ROOT / "submission" / f"lab3_{args.cnetid}.zip"
STAGE.parent.mkdir(exist_ok=True)

def local(p):  # records store absolute paths under either the old symlink or the real path
    p = str(p)
    for pre in ("/Users/tusharsonthalia/cmsc35200/lab3/", str(ROOT) + "/",
                "/Users/tusharsonthalia/Documents/Masters/UChicago/Academics/Q4/#5. Deep Learning Systems/Lab3/lab3/"):
        if p.startswith(pre): return ROOT / p[len(pre):]
    return Path(p)

rs = [json.loads(l) for l in open(ROOT / "results/runs.jsonl") if l.strip()]
ann = (ROOT / "results/ANNOTATIONS.md").read_text()
sel_dir = ROOT / "checkpoints/selected"; sel_dir.mkdir(parents=True, exist_ok=True)
index = ["# Selected compressed exports", "",
         "Copies of the selected compressed exports. The originals sit in the large pilot and artifact folders,",
         "which are not submitted. Each row names the run_id that produced the file.", "",
         "| file | part | what | run_id |", "|---|---|---|---|"]

def take(src, what, part, run_id):
    src = local(src)
    if not src.exists():  # originals may have been archived; keep an existing copy
        status = "" if (sel_dir / src.name).exists() else "(missing) "
        index.append(f"| {status}{src.name} | {part} | {what} | {run_id} |"); return
    shutil.copy2(src, sel_dir / src.name); index.append(f"| {src.name} | {part} | {what} | {run_id} |")

# Part B: the 12 confirmed points, seed-0 pilot post_finetune rows
for s in json.load(open(ROOT / "results/confirm_selection.json")):
    r = next(r for r in rs if r["run_id"] == s["selected_run_id"])
    take(r["checkpoint"], f"{s['arch']} {s['method']} t={s['selected_target']} masked/rebuilt checkpoint", "B", r["run_id"])
    comp = ((r.get("paths") or {}).get("compact") or {}).get("artifact")
    if comp: take(comp, f"{s['arch']} {s['method']} t={s['selected_target']} compact export", "B", r["run_id"])

# Part C: latest ok row per (phase, arch, method) on the pinned Python, not superseded
latest = {}
for r in sorted((r for r in rs if r.get("part") == "C" and r["status"] == "ok"
                 and r.get("phase") in ("precision", "factorial")
                 and r["versions"]["python"] == "3.12.13"), key=lambda r: r["timestamp"]):
    latest[(r["phase"], r["architecture"], r["method"])] = r
for (ph, a, m), r in sorted(latest.items()):
    if r.get("artifact"): take(r["artifact"], f"{a} {m} ({ph})", "C", r["run_id"])
(sel_dir / "INDEX.md").write_text("\n".join(index) + "\n")

# Stage
INCLUDE = ["README.md", "memo.pdf", "predictions.md", "AGENT_DISCLOSURE.md", " ",
           "pyproject.toml", "uv.lock", ".python-version", "code", "tests", "configs", "splits",
           "scripts", "results", "checkpoints/dense", "checkpoints/selected", "figures", "transcripts"]
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "compacts", "precision_artifacts"}
EXCLUDE_SUFFIX = {".pyc"}
if STAGE.exists(): shutil.rmtree(STAGE)
files = []
for item in INCLUDE:
    src = ROOT / item
    if not src.exists():
        print(f"WARNING: {item} missing" + (" — add your memo.pdf, then rerun" if item == "memo.pdf" else "")); continue
    paths = [src] if src.is_file() else [p for p in src.rglob("*") if p.is_file()]
    for p in paths:
        rel = p.relative_to(ROOT)
        if any(part in EXCLUDE_DIRS for part in rel.parts) or p.suffix in EXCLUDE_SUFFIX or p.name == ".DS_Store": continue
        if rel.parts[:2] == ("results", "bench") and p.suffix == ".pt": continue
        dst = STAGE / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst); files.append(rel)

# Manifest
lines = []
for rel in sorted(files, key=str):
    h = hashlib.sha256((STAGE / rel).read_bytes()).hexdigest(); lines.append(f"{h}  {rel}")
(STAGE / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")

# Zip
if ZIP.exists(): ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted(STAGE.rglob("*")):
        if p.is_file(): z.write(p, Path(STAGE.name) / p.relative_to(STAGE))
print(f"staged {len(files)} files + MANIFEST.sha256 -> {STAGE}")
print(f"zip {ZIP} ({ZIP.stat().st_size/1e6:.1f} MB)")
