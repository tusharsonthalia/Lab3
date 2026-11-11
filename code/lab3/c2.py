"""Part C (reduced precision) C2 drivers.

Moved into the package from the standalone ``code/lab3_precision.py`` module
(deleted per C2 item 7) so the study is driven via:

    uv run python -m lab3 precision
    uv run python -m lab3 factorial

Per predictions.md C.1 / C.2 / C.3:
  results/runs.jsonl          phase="precision" rows (5 formats x 3 archs)
  results/int8_selection.json INT8 method selection per architecture
  results/factorial.csv       12-condition factorial
  results/precision_artifacts/<arch>-<method>_<run_id>.pt
  results/partc_summary.md    (written by the summary step)
"""
from __future__ import annotations

import json

import torch

from .util import RESULTS_DIR, code_hash, set_threads
from .evaln import load_checkpoint
from .prune import PILOT_DIR
from . import precision as P

ARCHS = ("mlp", "cnn", "gcn")
FORMATS = ("fp16", "bf16", "int8_dynamic", "int8_static", "int4_simulated")


def partc_selection() -> dict:
    p = RESULTS_DIR / "partc_selection.json"
    return json.load(open(p))


def _parent_run_id(arch: str) -> str:
    """Look up the seed-0 dense Part A run id for the architecture."""
    runs = [json.loads(ln) for ln in (RESULTS_DIR / "runs.jsonl").read_text().splitlines() if ln.strip()]
    for r in runs:
        if (r.get("architecture") == arch
                and r.get("seed") == 0
                and r.get("method") == "dense"
                and (r.get("part") == "A" or r.get("phase") is None)
                and r.get("val_acc") is not None):
            return r["run_id"]
    return "dense_seed0"


def dense_parent(arch: str) -> torch.nn.Module:
    parent, _ = load_checkpoint(arch, 0)
    parent.eval()
    return parent


def run_precision(arch: str | None = None) -> list:
    archs = (arch,) if arch else ARCHS
    out = []
    set_threads()
    P.set_qnnpack_engine()
    for a in archs:
        parent = dense_parent(a)
        prid = _parent_run_id(a)
        for fmt in FORMATS:
            print(f"[precision] {a} {fmt}", flush=True)
            rec = P.convert_and_measure(a, fmt, parent, prid, write=True)
            print(f"   status={rec.get('status')} val_acc={rec.get('val_acc')} "
                  f"nonzero={rec.get('nonzero_params')} "
                  f"frac_macs={rec.get('frac_macs_reduced_precision')} "
                  f"artifact_bytes={rec.get('artifact_bytes')} "
                  f"reload={rec.get('reload_equivalence')}", flush=True)
            out.append(rec)
    return out


def run_int8_selection(arch: str | None = None) -> dict:
    archs = (arch,) if arch else ARCHS
    out = {}
    set_threads()
    P.set_qnnpack_engine()
    for a in archs:
        parent = dense_parent(a)
        prid = _parent_run_id(a)
        print(f"[int8_selection] {a}", flush=True)
        dyn_rec = P.convert_and_measure(a, "int8_dynamic", parent, prid, write=False)
        stat_rec = P.convert_and_measure(a, "int8_static", parent, prid, write=False)
        out[a] = P.select_int8(a, parent, prid, dyn_rec, stat_rec)
        print(f"   selected={out[a].get('selected')} reason={out[a].get('reason')}", flush=True)
    return out


def run_int8_rerun(arch: str | None = None) -> dict:
    """C3: re-run ONLY the INT8 records as new rows (the C2 int8 rows are
    superseded because their integer-weight accounting was blind to the
    quantized projection weights). Re-runs:
      - standalone int8_dynamic + int8_static for each arch (phase="precision");
      - dense_int8 + sparse_int8 factorial conditions for each arch
        (phase="factorial", using the C2-selected int8 method per arch from
        int8_selection.json).
    Returns {"precision": [...], "factorial": [...], "selection": {...}}.
    The selection is recomputed from the NEW precision rows; the caller writes
    it to results/int8_selection.json.
    """
    archs = (arch,) if arch else ARCHS
    prec_out = []
    fact_out = []
    sel_out = {}
    set_threads()
    P.set_qnnpack_engine()
    sel_path = RESULTS_DIR / "int8_selection.json"
    for a in archs:
        parent = dense_parent(a)
        prid = _parent_run_id(a)
        # Standalone int8_dynamic + int8_static (phase="precision").
        dyn_rec = P.convert_and_measure(a, "int8_dynamic", parent, prid, write=True)
        stat_rec = P.convert_and_measure(a, "int8_static", parent, prid, write=True)
        prec_out.extend([dyn_rec, stat_rec])
        # Recompute selection from the new precision rows.
        sel_out[a] = P.select_int8(a, parent, prid, dyn_rec, stat_rec)
        int8_method = sel_out[a].get("selected", "int8_dynamic")
        quant = (P.int8_quantize_dynamic if int8_method == "int8_dynamic"
                 else P.int8_quantize_static)
        # Factorial dense_int8 + sparse_int8 (phase="factorial").
        pcsel = partc_selection()[a]
        sparse_arm = pcsel["selected"]["pilot_arm_id"]
        sparse_model, _ = _load_sparse_checkpoint(a, sparse_arm)
        sparse_model.eval()
        dense_q, _ = quant(parent, a)
        sparse_q, _ = quant(sparse_model, a)
        dense_nn = sum(int(p.count_nonzero()) for p in parent.parameters())
        sparse_nn = sum(int(p.count_nonzero()) for p in sparse_model.parameters())
        for cond, model, nnz, src in (("dense_int8", dense_q, dense_nn, parent),
                                      ("sparse_int8", sparse_q, sparse_nn, sparse_model)):
            print(f"[int8_rerun] {a} {cond} ({int8_method})", flush=True)
            rec = P.factorial_row(a, cond, model, prid, int8_method, src,
                                  write=True, nonzero_count=nnz, src_fp32=parent)
            print(f"   status={rec.get('status')} val_acc={rec.get('val_acc')} "
                  f"nonzero_after_int8={rec.get('nonzero_after_int8')} "
                  f"artifact_bytes={rec.get('artifact_bytes')} "
                  f"bytes_by_dtype={rec.get('bytes_by_dtype')}", flush=True)
            fact_out.append(rec)
    return {"precision": prec_out, "factorial": fact_out, "selection": sel_out}


def run_c4_artifact_rerun(arch: str | None = None) -> dict:
    """C4: re-run ONLY the INT8 records as new rows because the C3 artifacts
    used a nested-pickle container for mlp/gcn INT8 (state =
    {"pickled": <nested torch.save(model) zip bytes>}). The C4 artifact format
    is a single torch.save of the converted model's state_dict (int8 weights +
    scales + zero points; no nested pickled bytes, no fp32 weight copy),
    reloaded by rebuilding the same quantized structure from the fp32 parent
    and load_state_dict(strict=True) with bit-identical logits.

    Re-runs the same 12 rows as run_int8_rerun:
      - standalone int8_dynamic + int8_static per arch (phase="precision");
      - dense_int8 + sparse_int8 factorial conditions per arch, using the
        selected INT8 method from int8_selection.json (phase="factorial").
    The selection is recomputed from the NEW precision rows; the caller writes
    it to results/int8_selection.json (supersedes field points to the C3
    rows).
    """
    archs = (arch,) if arch else ARCHS
    prec_out = []
    fact_out = []
    sel_out = {}
    set_threads()
    P.set_qnnpack_engine()
    # C3 rows we are superseding: the latest ok int8 C-rows that carry a
    # `supersedes` field (i.e. the C3 re-run rows that produced the
    # nested-pickle artifacts).
    runs = [json.loads(ln) for ln in (RESULTS_DIR / "runs.jsonl").read_text()
            .splitlines() if ln.strip()]
    c3: dict[tuple[str, str], str] = {}
    for r in runs:
        if (r.get("part") == "C" and r.get("status") == "ok"
                and r.get("supersedes") is not None
                and (r.get("method") in ("int8_dynamic", "int8_static", "dense_int8", "sparse_int8"))):
            c3[(r.get("architecture"), r.get("method"))] = r["run_id"]
    sel_path = RESULTS_DIR / "int8_selection.json"
    sel_prev: dict = {}
    if sel_path.exists():
        sel_prev = json.load(open(sel_path)).get("selection", {})
    for a in archs:
        parent = dense_parent(a)
        prid = _parent_run_id(a)
        dyn_rec = P.convert_and_measure(a, "int8_dynamic", parent, prid, write=False)
        stat_rec = P.convert_and_measure(a, "int8_static", parent, prid, write=False)
        sel_out[a] = P.select_int8(a, parent, prid, dyn_rec, stat_rec)
        # Persist the new precision rows with supersedes -> C3 rows.
        for rec in (dyn_rec, stat_rec):
            prev = c3.get((a, rec["method"]))
            if prev:
                rec["supersedes"] = prev
            P.write_run_record(rec)
            prec_out.append(rec)
        int8_method = sel_out[a].get("selected", "int8_dynamic")
        quant = (P.int8_quantize_dynamic if int8_method == "int8_dynamic"
                 else P.int8_quantize_static)
        pcsel = partc_selection()[a]
        sparse_arm = pcsel["selected"]["pilot_arm_id"]
        sparse_model, _ = _load_sparse_checkpoint(a, sparse_arm)
        sparse_model.eval()
        dense_q, _ = quant(parent, a)
        sparse_q, _ = quant(sparse_model, a)
        dense_nn = sum(int(p.count_nonzero()) for p in parent.parameters())
        sparse_nn = sum(int(p.count_nonzero()) for p in sparse_model.parameters())
        for cond, model, nnz, src in (("dense_int8", dense_q, dense_nn, parent),
                                      ("sparse_int8", sparse_q, sparse_nn, sparse_model)):
            print(f"[c4_rerun] {a} {cond} ({int8_method})", flush=True)
            rec = P.factorial_row(a, cond, model, prid, int8_method, src,
                                  write=False, nonzero_count=nnz, src_fp32=parent)
            prev = c3.get((a, cond))
            if prev:
                rec["supersedes"] = prev
            P.write_run_record(rec)
            print(f"   status={rec.get('status')} val_acc={rec.get('val_acc')} "
                  f"artifact_bytes={rec.get('artifact_bytes')} "
                  f"bytes_by_dtype={rec.get('bytes_by_dtype')} "
                  f"reload={rec.get('reload_equivalence')}", flush=True)
            fact_out.append(rec)
    # Merge selection into the file (keep all three archs; update code hash).
    merged: dict = dict(sel_prev)
    merged.update(sel_out)
    with open(sel_path, "w") as f:
        json.dump({"code": P.code_hash(), "selection": merged}, f, indent=1)
    print(f"[c4_rerun] wrote {sel_path}", flush=True)
    return {"precision": prec_out, "factorial": fact_out, "selection": sel_out}


def _load_sparse_checkpoint(arch: str, pilot_arm_id: str) -> tuple[torch.nn.Module, dict]:
    """Load the sparse seed-0 pilot checkpoint selected by partc_selection.json."""
    path = PILOT_DIR / f"{pilot_arm_id}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    m = P.build_model(arch)
    m.load_state_dict(payload["state_dict"])
    m.eval()
    return m, payload


def run_factorial(arch: str | None = None) -> list[dict]:
    archs = (arch,) if arch else ARCHS
    set_threads()
    P.set_qnnpack_engine()
    sel = partc_selection()
    rows = []
    for a in archs:
        parent = dense_parent(a)
        prid = _parent_run_id(a)
        sparse_arm = sel[a]["selected"]["pilot_arm_id"]
        sparse_model, _ = _load_sparse_checkpoint(a, sparse_arm)
        sparse_model.eval()
        # The INT8 method to apply is the one SELECTED by the frozen C.2 rule
        # (recomputed from the new precision rows -> results/int8_selection.json).
        import json as _json
        sel_path = RESULTS_DIR / "int8_selection.json"
        if sel_path.exists():
            with open(sel_path) as f:
                sel2 = _json.load(f)["selection"]
            int8_method = sel2[a]["selected"]
        else:
            int8_method = "int8_dynamic"
        quant = (P.int8_quantize_dynamic if int8_method == "int8_dynamic"
                 else P.int8_quantize_static)
        dense_q, _ = quant(parent, a)
        sparse_q, _ = quant(sparse_model, a)
        dense_nn = sum(int(p.count_nonzero()) for p in parent.parameters())
        sparse_nn = sum(int(p.count_nonzero()) for p in sparse_model.parameters())
        conditions = [
            ("dense_fp32", parent, "fp32", dense_nn, parent),
            ("sparse_fp32", sparse_model, "fp32", sparse_nn, sparse_model),
            ("dense_int8", dense_q, int8_method, dense_nn, parent),
            ("sparse_int8", sparse_q, int8_method, sparse_nn, sparse_model),
        ]
        for cond, model, kind, nnz, src in conditions:
            print(f"[factorial] {a} {cond} (int8={int8_method})", flush=True)
            model.eval()
            rec = P.factorial_row(a, cond, model, prid, kind, parent,
                                  write=True, nonzero_count=nnz, src_fp32=src)
            rows.append(rec)
    return rows


def run_int8_selection_and_factorial(arch: str | None = None) -> None:
    """INT8 selection (writes results/int8_selection.json) + 12-condition
    factorial (writes results/factorial.csv). Both run from the C2 precision
    rows; the INT8 method applied to the factorial is the selected one."""
    sel = run_int8_selection(arch)
    archs = (arch,) if arch else ARCHS
    # Merge into any existing selection file (keeps all three archs if a
    # single arch is rerun).
    sel_path = RESULTS_DIR / "int8_selection.json"
    merged: dict = {}
    if sel_path.exists():
        try:
            merged = json.load(open(sel_path)).get("selection", {})
        except Exception:
            merged = {}
    merged.update(sel)
    with open(sel_path, "w") as f:
        json.dump({"code": code_hash(), "selection": merged}, f, indent=1)
    print(f"[int8_selection] wrote {sel_path}", flush=True)

    rows = run_factorial(arch)
    _write_factorial_csv(rows)
    print(f"[factorial] wrote {len(rows)} rows to results/factorial.csv", flush=True)


def _write_factorial_csv(rows: list[dict]) -> None:
    import csv
    fieldnames = ["part", "phase", "method", "architecture", "val_acc", "test_acc",
                  "total_params", "nonzero_params", "nonzero_before_int8",
                  "nonzero_after_int8", "frac_macs_reduced_precision",
                  "reduced_precision_ops", "artifact_bytes", "bytes_by_dtype",
                  "reload_equivalence", "latency", "status", "run_id"]
    with open(RESULTS_DIR / "factorial.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["bytes_by_dtype"] = json.dumps(rr.get("bytes_by_dtype"))
            rr["reduced_precision_ops"] = "|".join(rr.get("reduced_precision_ops") or [])
            w.writerow(rr)


def _write_container_overhead_md(rows: list[dict]) -> None:
    """C3: write results/artifact_zip_listing.md — per-entry zipfile.infolist()
    for the 12 latest-ok factorial artifacts, from each row's
    `container_overhead` field (so it reflects the actual saved files)."""
    out = []
    out.append("# Factorial artifact zip listings (C4)\n")
    out.append("Per-entry `zipfile.ZipFile(artifact).infolist()` for the 12 latest-ok\n")
    out.append("3.12 factorial artifacts. `file_size` = stored (uncompressed) bytes;\n")
    out.append("`compress_size` = bytes on disk; `compress_type` 0 = stored (no\n")
    out.append("compression). All entries are the outer .pt zip (torch.save of\n")
    out.append("`{kind, architecture, state}`). Since C4 every `state` is a flat\n")
    out.append("state dict (int8: the converted model's state_dict with int8\n")
    out.append("weights + scales + zero points; fp32: the fp32 state dict), so\n")
    out.append("there is NO inner/nested zip. See C4_DONE.md for the fix of the\n")
    out.append("C3 nested-pickle artifact container.\n")
    for arch in ("mlp", "cnn", "gcn"):
        arch_rows = [r for r in rows if r.get("architecture") == arch]
        for cond in ("dense_fp32", "sparse_fp32", "dense_int8", "sparse_int8"):
            row = next((r for r in arch_rows if r.get("method") == cond and r.get("status") == "ok"), None)
            if row is None:
                continue
            co = row.get("container_overhead")
            # C3: the fp32 rows were written by C2 (no container_overhead field).
            # Compute it on the fly from the artifact path so the listing is
            # complete for all 12 conditions.
            if co is None and row.get("artifact"):
                try:
                    from .precision import _container_overhead
                    co = _container_overhead(row["artifact"])
                except Exception:
                    co = {}
            co = co or {}
            out.append(f"## {arch} {cond}  ({row.get('run_id')})\n")
            out.append(f"artifact_bytes = {co.get('artifact_bytes')}; "
                       f"payload_bytes = {row.get('payload_bytes')}; "
                       f"outer structural = {co.get('structural_bytes')}; "
                       f"outer stored = {co.get('stored_bytes')}\n")
            out.append("| entry | file_size | compress_size | compress_type |")
            out.append("|---|---|---|---|")
            for e in co.get("entries", []):
                out.append(f"| OUT {e['name']} | {e['file_size']} | {e['compress_size']} | {e['compress_type']} |")
            out.append("")
    (RESULTS_DIR / "artifact_zip_listing.md").write_text("\n".join(out) + "\n")
    print(f"[summary] wrote {RESULTS_DIR / 'artifact_zip_listing.md'}", flush=True)


def write_summary() -> None:
    """Write results/partc_summary.md from the C2 (python 3.12) run rows."""
    runs = [json.loads(ln) for ln in (RESULTS_DIR / "runs.jsonl").read_text().splitlines() if ln.strip()]

    def c2(r: dict) -> bool:
        py = (r.get("versions") or {}).get("python", "")
        return py.startswith("3.12")

    prec_rows: dict[str, list[dict]] = {}
    fact_rows: dict[str, list[dict]] = {}
    for r in runs:
        if r.get("part") != "C" or not c2(r):
            continue
        if r.get("phase") == "precision":
            prec_rows.setdefault(r.get("architecture"), []).append(r)
        if r.get("phase") == "factorial":
            fact_rows.setdefault(r.get("architecture"), []).append(r)

    # latest ok row per (arch, fmt) for the standalone table. File order in
    # runs.jsonl == write order, so "last ok wins" naturally picks the C3 int8
    # rows over the superseded C2 int8 rows (C3 is written after C2).
    def latest_ok(rows: list[dict]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in rows:
            if r.get("status") == "ok":
                out[r["method"]] = r  # last ok per method wins
        return out

    out = []
    out.append("# Part C (C4) summary\n")
    out.append("Reduced precision: fp16, bf16, dynamic-INT8 (qnnpack), static-INT8 (qnnpack FX), "
               "simulated-INT4, 5 formats x 3 architectures, run on the pinned Python 3.12 "
               "environment (torch 2.15.0). This supersedes the C1b run (Python 3.13 env mismatch) "
               "and the C2 run (2026-11-09/10); the C2 INT8 rows are superseded by C3 because the "
               "C2 integer-weight accounting (`nonzero_after_int8`, `bytes_by_dtype`) was blind to the "
               "quantized projection weights (it only matched modules with a working `_weight_bias`, "
               "missing the FX-static `_packed_weight_N` / raw-ScriptObject shapes — GCN static int8 "
               "reported nonzero_after_int8=23 and bytes_by_dtype={fp32:92}); the C3 INT8 rows are "
               "superseded by C4 because the C3 INT8 artifacts for mlp/gcn used a nested-pickle "
               "container (`state = {\"pickled\": <nested torch.save(model) zip bytes>}`), which "
               "re-encoded the whole module tree inside the outer zip instead of storing a plain "
               "state dict. All 45 C1b, the C2 part=\"C\" rows, and the 12 C3 INT8 rows are marked "
               "superseded in results/ANNOTATIONS.md. Fields: artifact_bytes "
               "+ bytes_by_dtype (per-stored-dtype payload, container overhead = artifact_bytes - "
               "payload_bytes; per-entry zip breakdown in results/artifact_zip_listing.md), "
               "nonzero_params (integer-representation nonzero count), frac_macs_reduced_precision, "
               "latency (batch 1 + batch 256 / full graph, median + IQR), rss, "
               "stored/compute/accumulator precision, reload_equivalence, run_id.\n")
    out.append("C3 also corrects the C2 explanation of the `pip install -e .` failure: the real cause "
               "was that `uv venv --seed` writes the site-packages `.pth` files with the macOS "
               "`UF_HIDDEN` (0x80000000) file flag, and CPython's `site.py` (3.12) skips hidden `.pth` "
               "files, so `lab3.pth` was ignored and `import lab3` failed — NOT the `#`-comment issue "
               "C2_DONE.md described. (C2_DONE.md is immutable; the correction lives here and in "
               "SCHEMA.md's C3 changelog.)\n")

    out.append("## Standalone conversions (phase=\"precision\")\n")
    out.append("| arch | format | val_acc | total_params | nonzero_params | nonzero(dense parent) | "
               "frac_macs | stored | compute | accumulator | artifact_bytes | payload_bytes | bytes_by_dtype | reload max_diff | status | run_id |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arch in ARCHS:
        by_fmt = latest_ok(prec_rows.get(arch, []))
        for fmt in FORMATS:
            r = by_fmt.get(fmt)
            if r is None:
                out.append(f"| {arch} | {fmt} | — (no ok C2 row) | | | | | | | | | | | | | |")
                continue
            re = r.get("reload_equivalence") or {}
            bbd = r.get("bytes_by_dtype") or {}
            out.append(
                f"| {arch} | {fmt} | {r.get('val_acc')} | {r.get('total_params')} | "
                f"{r.get('nonzero_params')} | {r.get('nonzero_params_dense_parent')} | "
                f"{r.get('frac_macs_reduced_precision')} | {r.get('stored_precision')} | "
                f"{r.get('compute_precision')} | {r.get('accumulator_precision')} | "
                f"{r.get('artifact_bytes')} | "
                f"{r.get('payload_bytes')} | {bbd} | {re.get('max_abs_diff')} | "
                f"{r.get('status')} | {r.get('run_id')} |"
            )

    out.append("\n## Latency (median_ms / IQR_ms) and RSS (delta bytes)\n")
    out.append("| arch | format | bs1 med | bs1 IQR | bs256/full med | bs256/full IQR | rss_delta |")
    out.append("|---|---|---|---|---|---|---|")
    for arch in ARCHS:
        by_fmt = latest_ok(prec_rows.get(arch, []))
        lkey = "full_graph" if arch == "gcn" else "batch_256"
        bkey = "batch_1"
        for fmt in FORMATS:
            r = by_fmt.get(fmt)
            if r is None:
                continue
            lat = r.get("latency") or {}
            b1 = lat.get(bkey) or {}
            b2 = lat.get(lkey) or {}
            out.append(
                f"| {arch} | {fmt} | {b1.get('median_ms')} | {b1.get('iqr_ms')} | "
                f"{b2.get('median_ms')} | {b2.get('iqr_ms')} | "
                f"{r.get('model_rss_delta_bytes')} |"
            )

    # INT8 selection
    sel_path = RESULTS_DIR / "int8_selection.json"
    out.append("\n## INT8 selection (C.2 rule, C4 run)\n")
    out.append("| arch | selected | val_acc | latency_ms | reason | run_id |")
    out.append("|---|---|---|---|---|---|")
    if sel_path.exists():
        sel = json.load(open(sel_path)).get("selection", {})
        for arch in ARCHS:
            s = sel.get(arch, {})
            out.append(f"| {arch} | {s.get('selected')} | {s.get('val_acc')} | "
                       f"{s.get('latency_ms')} | {s.get('reason')} | {s.get('run_id')} |")

    # Factorial — latest ok per (arch, condition) (file order: C4 int8 rows win)
    out.append("\n## 12-condition factorial (phase=\"factorial\", C4 run)\n")
    out.append("| arch | condition | val_acc | total_params | nonzero | nonzero_before_int8 | nonzero_after_int8 | "
               "frac_macs | artifact_bytes | payload_bytes | container_overhead_bytes | reload max_diff | status | run_id |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arch in ARCHS:
        # latest ok per condition
        latest: dict[str, dict] = {}
        for r in fact_rows.get(arch, []):
            if r.get("status") == "ok":
                latest[r.get("method")] = r
        for cond in ("dense_fp32", "sparse_fp32", "dense_int8", "sparse_int8"):
            r = latest.get(cond)
            if r is None:
                continue
            re = r.get("reload_equivalence") or {}
            co = r.get("container_overhead") or {}
            ov = (co.get("artifact_bytes", 0) - (r.get("payload_bytes") or 0)) if co else ""
            out.append(
                f"| {arch} | {r.get('method')} | {r.get('val_acc')} | {r.get('total_params')} | "
                f"{r.get('nonzero_params')} | {r.get('nonzero_before_int8')} | "
                f"{r.get('nonzero_after_int8')} | {r.get('frac_macs_reduced_precision')} | "
                f"{r.get('artifact_bytes')} | {r.get('payload_bytes')} | {ov} | "
                f"{re.get('max_abs_diff')} | {r.get('status')} | {r.get('run_id')} |"
            )

    # Container-overhead explanation (C4: nested-pickle container fixed)
    out.append("\n## Container overhead: why the C3 int8 files were bloated (C4 fix)\n")
    out.append("`container_overhead_bytes = artifact_bytes - payload_bytes`. Per-entry breakdowns "
               "(zipfile.infolist) are in results/artifact_zip_listing.md.\n")
    out.append("The C3 explanation of the dense-vs-sparse INT8 file-size gap was WRONG: it blamed "
               "the gap on larger fp32 bias tensors in the nested inner zip, but the MLP biases total "
               "only ~2 KB, not ~99 KB. The real cause was the C3 INT8 artifact FORMAT: for mlp/gcn "
               "INT8 (dynamic and static) the saved file was a dict whose `state` was "
               "`{\"pickled\": <bytes of an inner torch.save of the whole live module>}` — a nested "
               "container. The inner zip re-encoded the ENTIRE module tree: every quantized weight "
               "storage appeared twice (once as the quantized weight, once via the module's fp32 "
               "placeholder/structure re-pickled inside the module's own state), plus duplicated "
               "module-structure pickles and a full second zip of headers. A plain `torch.save(q."
               "state_dict())` of the same converted model is 241,351 bytes for the dense AND the "
               "S1-90% sparse MLP (zeros cost 1 byte in int8, as expected), versus 359,751 (dense) / "
               "260,237 (sparse) for the nested-pickle C3 artifacts.\n")
    out.append("C4 fix: every INT8 artifact (dynamic and static, all 3 archs, standalone and "
               "factorial) is now a single torch.save of the converted model's `state_dict()` — no "
               "nested pickled bytes, no fp32 copy of the weights. Reload rebuilds the same "
               "quantized structure from the fp32 parent architecture (same quantize_dynamic / FX "
               "prepare+convert / GCN per-projection path) and calls `load_state_dict(strict=True)`; "
               "reloaded logits match the in-memory converted model exactly (atol 0). The C4 "
               "overhead is a few KB (zip structural bytes), identical in shape for all conditions; "
               "the dense-vs-sparse int8 gap is now only the scale/zero-point bytes (a few dozen "
               "bytes). fp16/bf16/int4 artifacts were already flat (state dict / packed-buffer dict) "
               "and were left unchanged.\n")

    (RESULTS_DIR / "partc_summary.md").write_text("\n".join(out) + "\n")
    print(f"[summary] wrote {RESULTS_DIR / 'partc_summary.md'}", flush=True)

    # Build the latest-ok factorial rows for the zip listing (same selection as
    # the factorial table above).
    fact_final: dict[tuple[str, str], dict] = {}
    for arch in ARCHS:
        for r in fact_rows.get(arch, []):
            if r.get("status") == "ok":
                fact_final[(arch, r.get("method"))] = r
    fact_ordered = [fact_final[(arch, cond)] for arch in ARCHS
                    for cond in ("dense_fp32", "sparse_fp32", "dense_int8", "sparse_int8")
                    if (arch, cond) in fact_final]
    _write_container_overhead_md(fact_ordered)
