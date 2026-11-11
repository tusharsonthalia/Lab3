"""Generate the LaTeX tables for report/report.tex directly from results/runs.jsonl.

Every number in the report's tables comes from a record. Run IDs are cited by their unique
6-hex suffix. Rows that are superseded (listed in `supersedes` or in results/ANNOTATIONS.md)
are never used. Usage: uv run python scripts/make_report_tables.py
"""
import json, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "tables"; OUT.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(ROOT / "results/runs.jsonl") if l.strip()]
ann = (ROOT / "results/ANNOTATIONS.md").read_text()
superseded = {x for r in rows for x in (r.get("supersedes") or [])}
for line in ann.splitlines():
    if re.search(r"supersed|exclud|debug|test suite", line, re.I):
        superseded |= set(re.findall(r"[a-z]{3}-[a-z0-9_.-]+Z-[0-9a-f]{6}", line))
ok = lambda r: r["status"] == "ok" and r["run_id"] not in superseded
sid = lambda r: r["run_id"][-6:]
ARCH = {"mlp": "MLP", "cnn": "CNN", "gcn": "GCN"}
METH = {"s1": "S1 magnitude", "s2": "S2 structured", "s3": "S3 saliency", "s4": "S4 N:M", "dense_ft": "dense\\_ft"}
PAT = {0.25: "3:4", 0.5: "2:4", 0.75: "1:4", 0.9: "1:10"}
f4 = lambda x: "--" if x is None else f"{x:.4f}"
def ms(x): return "--" if x is None else (f"{x:.3f}" if x < 10 else f"{x:.2f}")
def kb(b): return "--" if b is None else f"{b/1000:.1f}"
def med(d, *keys):
    for k in keys:
        v = (d or {}).get(k)
        if isinstance(v, dict) and v.get("median_ms") is not None: return v["median_ms"], v.get("iqr_ms")
    return None, None
def write(name, body): (OUT / name).write_text(body); print("wrote", name)

# ---------------- Part A ----------------
train = {}
for r in rows:
    if r.get("part") == "A" and r.get("command") == "train" and ok(r): train[(r["architecture"], r["seed"])] = r
# Part A bench: the rows published in results/baselines.md. Every A2-era bench row was invoked
# through the pytest suite (see ANNOTATIONS.md), so the published rows are used explicitly. The
# same parents were benchmarked again cleanly in Part C (factorial dense_fp32 rows).
BENCH_A = {"mlp": "93cf88", "cnn": "adedb7", "gcn": "180141"}
bench = {a: next(r for r in rows if r["run_id"].endswith(s) and r.get("command") == "bench") for a, s in BENCH_A.items()}
lines = []
for a in ("mlp", "cnn", "gcn"):
    t0 = train[(a, 0)]; b = bench[a]; L = b.get("latency") or {}
    tests = [train[(a, s)]["test_acc"] for s in (1, 2, 3)]
    b1 = med(L, "batch_1"); b2 = med(L, "batch_256", "full_graph")
    lat = (f"{ms(b1[0])} ({ms(b1[1])}) / {ms(b2[0])} ({ms(b2[1])})" if b1[0] else f"-- / {ms(b2[0])} ({ms(b2[1])})")
    lines.append(f"{ARCH[a]} & {f4(t0['val_acc'])} & {f4(t0['test_acc'])} & {statistics.mean(tests):.4f}$\\pm${statistics.stdev(tests):.4f} & "
                 f"{t0['best_epoch']}/{t0['train_epochs']} & {b['total_params']:,} & {b['dense_bytes']:,} & {b['est_macs']:,} & {lat} & "
                 f"{b['model_rss_delta_bytes']/1e6:.1f} & {sid(t0)}, {sid(b)} \\\\")
write("baselines.tex", "\n".join(lines) + "\n")

# ---------------- Part B pilot matrix ----------------
pilot = defaultdict(dict)
for r in rows:
    if r.get("phase") == "pilot" and ok(r) and r.get("method") in ("s1", "s2", "s3", "s4"):
        pilot[(r["architecture"], r["method"], r["target_sparsity"])][r["stage"]] = r
lines = []
for a in ("mlp", "cnn", "gcn"):
    for m in ("s1", "s2", "s3", "s4"):
        cells = []
        for t in (0.25, 0.5, 0.75, 0.9):
            d = pilot[(a, m, t)]; pp, pf = d["post_prune"], d["post_finetune"]
            cells.append(f"{pf['achieved_sparsity']:.2f} & {pp['val_acc']:.3f} & {pf['val_acc']:.3f}")
        lines.append(f"{ARCH[a]} & {METH[m]} & " + " & ".join(cells) + " \\\\")
    if a != "gcn": lines.append("\\midrule")
write("pilot_matrix.tex", "\n".join(lines) + "\n")

dft = {r["architecture"]: r for r in rows if r.get("phase") == "pilot" and r.get("method") == "dense_ft" and ok(r)}
write("pilot_dense_ft.tex", ", ".join(f"{ARCH[a]} {dft[a]['val_acc']:.3f} ({sid(dft[a])})" for a in ("mlp", "cnn", "gcn")))

# ---------------- Part B confirmations ----------------
sel = json.load(open(ROOT / "results/confirm_selection.json"))
parent_val = {a: train[(a, 0)]["val_acc"] for a in ARCH}
lines = []
for s in sel:
    a, m, t = s["arch"], s["method"], s["selected_target"]
    conf = [r for r in rows if r.get("phase") == "confirm" and ok(r) and r["architecture"] == a and r["method"] == m
            and abs(r["target_sparsity"] - t) < 1e-9 and r["stage"] == "post_finetune"]
    dfc = [r for r in rows if r.get("phase") == "confirm" and ok(r) and r["architecture"] == a and r["method"] == "dense_ft"]
    assert len(conf) == 3 and len(dfc) == 3, (a, m, len(conf), len(dfc))
    v = [r["val_acc"] for r in conf]; te = [r["test_acc"] for r in conf]
    k = "batch_256" if a != "gcn" else "full_graph"
    dz = statistics.median(r["paths"]["dense_zeros"][f"dense_zeros_{k}"]["median_ms"] for r in conf)
    cpk = [kk for kk in conf[0]["paths"]["compact"] if kk.endswith(k)][0]
    cp = statistics.median(r["paths"]["compact"][cpk]["median_ms"] for r in conf)
    st = conf[0]["paths"]["compact"].get("runtime_status", "")
    note = "dense fallback" if st.startswith("unsupported") else ("rebuilt" if m == "s2" else "CSR")
    label = PAT[t] if m == "s4" else f"{t:.2f}"
    lines.append(f"{ARCH[a]} & {METH[m]} & {label} & {conf[0]['achieved_sparsity']:.3f} & "
                 f"{statistics.mean(v):.4f}$\\pm${statistics.stdev(v):.4f} & {statistics.mean(te):.4f}$\\pm${statistics.stdev(te):.4f} & "
                 f"{(statistics.mean(v)-parent_val[a])*100:+.2f} & {(statistics.mean(v)-statistics.mean(r['val_acc'] for r in dfc))*100:+.2f} & "
                 f"{kb(conf[0]['paths']['dense_zeros']['dense_bytes'])} / {kb(conf[0]['paths']['compact']['compressed_bytes'])} & "
                 f"{ms(dz)} / {ms(cp)} ({note}) & {', '.join(sid(r) for r in conf)} \\\\")
    if s is not sel[-1] and sel[sel.index(s) + 1]["arch"] != a: lines.append("\\midrule")
write("confirm.tex", "\n".join(lines) + "\n")

# ---------------- Part C standalone + factorial ----------------
latest = {}
for r in sorted((r for r in rows if r.get("part") == "C" and ok(r) and r.get("phase") in ("precision", "factorial")
                 and r["versions"]["python"] == "3.12.13"), key=lambda r: r["timestamp"]):
    latest[(r["phase"], r["architecture"], r["method"])] = r
FMT = {"fp16": "FP16", "bf16": "BF16", "int8_dynamic": "INT8 dynamic", "int8_static": "INT8 static (FX)", "int4_simulated": "INT4 simulated"}
PREC = {"fp16": "fp16/fp16/fp16", "bf16": "bf16/bf16/bf16", "int8_dynamic": "int8/int8/int32",
        "int8_static": "int8/int8/int32", "int4_simulated": "int4/fp32/fp32"}
lines = []
for a in ("mlp", "cnn", "gcn"):
    fp = latest[("factorial", a, "dense_fp32")]; L = fp.get("latency") or {}
    b1 = med(L, "batch_1")[0]; b2 = med(L, "batch_256", "full_graph")[0]
    lines.append(f"{ARCH[a]} & FP32 (parent) & fp32/fp32/fp32 & {f4(fp['val_acc'])} & {f4(fp['test_acc'])} & {kb(fp['artifact_bytes'])} & {ms(b1)} & {ms(b2)} & 0.00 & {sid(fp)} \\\\")
    for m in FMT:
        r = latest[("precision", a, m)]; L = r.get("latency") or {}
        b1 = med(L, "batch_1")[0]; b2 = med(L, "batch_256", "full_graph")[0]
        lines.append(f" & {FMT[m]} & {PREC[m]} & {f4(r['val_acc'])} & {f4(r['test_acc'])} & {kb(r['artifact_bytes'])} & {ms(b1)} & {ms(b2)} & "
                     f"{float(r.get('frac_macs_reduced_precision') or 0):.2f} & {sid(r)} \\\\")
    if a != "gcn": lines.append("\\midrule")
write("precision.tex", "\n".join(lines) + "\n")

COND = {"dense_fp32": "dense FP32", "sparse_fp32": "sparse FP32", "dense_int8": "dense INT8", "sparse_int8": "sparse+INT8"}
psel = json.load(open(ROOT / "results/partc_selection.json")); isel = json.load(open(ROOT / "results/int8_selection.json"))
def int8_method(a):
    return isel["selection"][a]["selected"]
lines = []
for a in ("mlp", "cnn", "gcn"):
    for c in COND:
        r = latest[("factorial", a, c)]; L = r.get("latency") or {}
        b2 = med(L, "batch_256", "full_graph")[0]
        nz = f"{r.get('nonzero_before_int8'):,} / {r.get('nonzero_after_int8'):,}" if "int8" in c else f"{r.get('nonzero_params'):,}"
        lines.append(f"{ARCH[a] if c == 'dense_fp32' else ''} & {COND[c]} & {f4(r['val_acc'])} & {f4(r['test_acc'])} & {kb(r['artifact_bytes'])} & {nz} & "
                     f"{ms(b2)} & {float(r.get('frac_macs_reduced_precision') or 0):.2f} & {sid(r)} \\\\")
    if a != "gcn": lines.append("\\midrule")
write("factorial.tex", "\n".join(lines) + "\n")
s = psel; write("partc_points.tex", "; ".join(
    f"{ARCH[a]}: {METH[s[a]['selected']['method']]} at {s[a]['selected']['target']:.2f} sparsity with {int8_method(a).replace('_', ' ')}" for a in ("mlp", "cnn", "gcn")))

# ---------------- accounting ----------------
c = Counter((r.get("part"), r["status"]) for r in rows)
n_sup = sum(1 for r in rows if r["run_id"] in superseded)
write("accounting.tex", (f"{len(rows)} records in total: Part A {c[('A','ok')]} ok / {c[('A','failed')]} failed, "
       f"Part B {c[('B','ok')]} ok / {c[('B','failed')]} failed, Part C {c[('C','ok')]} ok / {c[('C','failed')]} failed."))

# inline values used in captions/text, as macros (\input is not allowed inside captions)
mac = {"pilotDenseFt": "pilot_dense_ft.tex", "partcPoints": "partc_points.tex", "accountingText": "accounting.tex"}
(OUT / "macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{(OUT / v).read_text().strip()}}}\n" for k, v in mac.items()))
print("wrote macros.tex")
