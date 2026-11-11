"""Task B3 figures.

Reads results/runs.jsonl (phase=pilot rows, phase=confirm rows, and the
Part A dense seed-0 train/bench records) and writes:
  figures/acc_vs_sparsity.png
  figures/latency_vs_accuracy.png
  figures/size_vs_sparsity.png
  figures/CAPTIONS.md

Deterministic and re-runnable: reflects the current runs.jsonl contents.
Source CSVs are named in captions per figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "code"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = _REPO / "results"
FIGURES = _REPO / "figures"
RUNS = RESULTS / "runs.jsonl"

ARCHS = ("mlp", "cnn", "gcn")
METHODS = ("s1", "s2", "s3", "s4")
# Frozen bench protocol: batch 256 for MLP/CNN, full graph for GCN.
LAT_KEY = {"mlp": "batch_256", "cnn": "batch_256", "gcn": "full_graph"}
XLABEL = {
    "mlp": "achieved sparsity of eligible weights (fraction zero) [-]",
    "cnn": "achieved sparsity of eligible weights (fraction zero) [-]",
    "gcn": "achieved sparsity of eligible weights (fraction zero) [-]",
}
LAT_X = {
    "mlp": "median forward latency, batch 256 (ms, log scale)",
    "cnn": "median forward latency, batch 256 (ms, log scale)",
    "gcn": "median forward latency, full graph (ms, log scale)",
}

COLORS = {"s1": "tab:blue", "s2": "tab:orange", "s3": "tab:green", "s4": "tab:red"}
SHAPES = {"s1": "o", "s2": "s", "s3": "^", "s4": "D"}


def _loads() -> list[dict]:
    out = []
    for line in open(RUNS):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _rows(recs: list[dict], phase: str, stage: str, status: str = "ok") -> list[dict]:
    return [
        r
        for r in recs
        if r.get("part") == "B" and r.get("phase") == phase
        and r.get("stage") == stage and r.get("status") == status
    ]


def _latest_a(recs: list[dict], arch: str) -> tuple[dict | None, dict | None]:
    """Latest (file order) Part A train + bench seed-0 ok records."""
    tr = be = None
    for r in recs:
        if (
            r.get("part") == "A" and r.get("architecture") == arch
            and r.get("status") == "ok" and r.get("seed") == 0
        ):
            if r.get("command") == "train":
                tr = r
            if r.get("command") == "bench":
                be = r
    return tr, be


def _path_lat(rec: dict, path: str, arch: str) -> float | None:
    """Per-path latency: paths.<path>.<path>_<batch_256|full_graph>.median_ms
    (the dense_zeros / compact paths each carry their own timing block; the
    top-level `latency` dict holds only the dense_zeros numbers)."""
    p = (rec.get("paths") or {}).get(path)
    if not isinstance(p, dict):
        return None
    v = p.get(f"{path}_{LAT_KEY[arch]}")
    if isinstance(v, dict) and isinstance(v.get("median_ms"), (int, float)):
        return float(v["median_ms"])
    return None


def _dense_lat(bench_rec: dict | None, arch: str) -> float | None:
    """Dense seed-0 latency: top-level latency.<batch_256|full_graph>."""
    if bench_rec is None:
        return None
    v = (bench_rec.get("latency") or {}).get(LAT_KEY[arch])
    if isinstance(v, dict) and isinstance(v.get("median_ms"), (int, float)):
        return float(v["median_ms"])
    return None


def _h(color, label, **kw) -> "plt.Line2D":
    h = plt.Line2D([], [], color=color, **kw)
    h.set_label(label)
    return h


def _shared_legend(fig, handles: list, title: str, ncol: int, pad: float) -> None:
    leg = fig.legend(
        handles, [h.get_label() for h in handles],
        title=title, loc="upper center", bbox_to_anchor=(0.5, -pad),
        ncol=ncol, fontsize=8, title_fontsize=8, framealpha=0.9,
        handlelength=2.2, columnspacing=1.0,
    )
    # C1 review: verify the shared legend actually rendered (non-empty).
    assert leg.get_texts(), "shared legend rendered no text entries"
    return leg


def _tight_savefig(fig, path: Path, **kw) -> None:
    """Save with bbox_inches='tight' so the bottom shared legend is never
    clipped (C1 review), and disable minor tick labels on log axes to avoid
    overlapping labels (C1 review)."""
    for ax in fig.get_axes():
        for scale in ("x", "y"):
            axv = getattr(ax, f"get_{scale}scale", lambda: "linear")()
            if axv == "log":
                ax.minorticks_off()
    fig.savefig(path, bbox_inches="tight", **kw)


def main() -> None:
    recs = _loads()
    FIGURES.mkdir(parents=True, exist_ok=True)

    # dense seed-0 references (Part A train seed-0, latest; bench seed-0, latest)
    dense = {a: _latest_a(recs, a) for a in ARCHS}
    dense_val = {a: (dense[a][0] or {}).get("val_acc") for a in ARCHS}
    dense_test = {a: (dense[a][0] or {}).get("test_acc") for a in ARCHS}
    dense_lat = {a: _dense_lat(dense[a][1], a) for a in ARCHS}
    dense_bytes = {a: (dense[a][1] or {}).get("dense_bytes") for a in ARCHS}
    for a in ARCHS:
        for nm, v in (("dense_val", dense_val[a]), ("dense_test", dense_test[a]),
                      ("dense_lat", dense_lat[a]), ("dense_bytes", dense_bytes[a])):
            assert v is not None, f"missing Part A dense seed-0 {nm} for {a}"

    # pilot rows (seed 0)
    pilot_pr = _rows(recs, "pilot", "post_prune")
    pilot_ft = _rows(recs, "pilot", "post_finetune")

    # confirm rows: latest ok post_finetune per (arch, method, seed)
    conf_ft: dict[tuple, dict] = {}
    for r in _rows(recs, "confirm", "post_finetune"):
        conf_ft[(r["architecture"], r["method"], r["seed"])] = r

    captions: list[str] = []

    # ----------------------------------------------------------------- a
    # acc_vs_sparsity: 3 panels, one line per method, dashed post_prune,
    # solid post_finetune (seed-0 pilot); open P markers = confirm mean
    # +/- std (seeds 1-3); horizontal reference lines with labels that
    # live in the legend, not on the plot.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for idx, arch in enumerate(ARCHS):
        ax = axes[idx]
        for m in METHODS:
            pp = sorted(
                (r for r in pilot_pr if r["architecture"] == arch and r["method"] == m),
                key=lambda r: r["achieved_sparsity"] or 0.0,
            )
            pf = sorted(
                (r for r in pilot_ft if r["architecture"] == arch and r["method"] == m),
                key=lambda r: r["achieved_sparsity"] or 0.0,
            )
            if pp:
                ax.plot(
                    [r["achieved_sparsity"] for r in pp],
                    [r["test_acc"] for r in pp],
                    linestyle="--", color=COLORS[m], marker="o", ms=3, lw=1.2,
                )
            if pf:
                ax.plot(
                    [r["achieved_sparsity"] for r in pf],
                    [r["test_acc"] for r in pf],
                    linestyle="-", color=COLORS[m], marker="s", ms=3, lw=1.2,
                )
        # dense references: plain lines, labels go to the legend
        ax.axhline(dense_test[arch], color="black", lw=1.0)
        ax.axhline(dense_val[arch], color="gray", lw=1.0)
        dft = [r for r in pilot_ft
               if r["architecture"] == arch and r["method"] == "dense_ft"]
        if dft:
            ax.axhline(dft[0]["test_acc"], color="black", lw=1.0, ls=":")
        # confirmatory points: mean +/- std over seeds 1-3
        for m in METHODS:
            vals, sps = [], []
            for s in (1, 2, 3):
                r = conf_ft.get((arch, m, s))
                if r is not None and r.get("test_acc") is not None:
                    vals.append(r["test_acc"])
                    sps.append(r["achieved_sparsity"])
            if vals:
                mean, x = float(np.mean(vals)), float(np.mean(sps))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                ax.errorbar(
                    [x], [mean], yerr=[std], fmt="P", ms=7, capsize=3,
                    color=COLORS[m], ecolor=COLORS[m], mfc="none", mew=1.5,
                )
        ax.set_title(arch.upper())
        ax.set_xlabel(XLABEL[arch])
        if idx == 0:
            ax.set_ylabel("test accuracy [-]")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    _shared_legend(
        fig,
        [_h(COLORS[m], f"{m} (color = method)", lw=1.4) for m in METHODS]
        + [
            _h("0.3", "dashed = post_prune (pilot seed-0)", lw=1.2, ls="--"),
            _h("0.3", "solid = post_finetune (pilot seed-0)", lw=1.2),
            _h("0.3", "open marker = confirm mean +/- std (seeds 1-3)", marker="P", ls="", mfc="none", ms=8),
            _h("black", "dense seed-0: black = test, gray = val (per panel)", lw=1.0),
            _h("black", "dense_ft seed-0 (dotted)", lw=1.0, ls=":"),
        ],
        title="references", ncol=3, pad=0.14,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _tight_savefig(fig, FIGURES / "acc_vs_sparsity.png", dpi=150)
    captions.append(
        "fig acc_vs_sparsity.png: 3 panels (MLP, CNN, GCN). x: achieved sparsity of "
        "eligible weights (fraction zero, -); y: test accuracy (-). One line per method "
        "(color = method): dashed = post_prune, solid = post_finetune, seed-0 pilot "
        "(results/pilot_summary.csv, phase=pilot rows). Open P markers: confirmatory "
        "points, mean +/- sample std of test_acc over seeds 1-3 "
        "(results/confirm_summary.csv, stage=post_finetune). Horizontal reference lines "
        "(labeled in the shared legend, not on the plot): black = dense seed-0 test acc, "
        "gray = dense seed-0 val acc (results/baselines.md, Part A train seed-0); dotted "
        "black = dense_ft seed-0 (pilot). Source: results/pilot_summary.csv, "
        "results/confirm_summary.csv, results/baselines.md."
    )
    plt.close(fig)

    # ----------------------------------------------------------------- b
    # latency_vs_accuracy: 3 panels, x = per-path latency median (log),
    # y = post_finetune test_acc, marker shape = method, hollow =
    # dense_zeros, filled = compact, for every pilot post_finetune arm.
    n_pts_per_panel = 0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for idx, arch in enumerate(ARCHS):
        ax = axes[idx]
        pts = 0
        for r in pilot_ft:
            if r["architecture"] != arch or r["method"] == "dense_ft":
                continue
            m = r["method"]
            assert m in SHAPES, f"unknown method {m}"
            assert r.get("test_acc") is not None, f"missing test_acc: {r['run_id']}"
            for path, fill in (("dense_zeros", False), ("compact", True)):
                lat = _path_lat(r, path, arch)
                assert lat is not None and lat > 0, (
                    f"missing {path} {LAT_KEY[arch]} median_ms for {arch}/{m}/t"
                    f"{r['target_sparsity']}: {r['run_id']}"
                )
                ax.scatter(
                    [lat], [r["test_acc"]],
                    marker=SHAPES[m], s=55,
                    facecolors=COLORS[m] if fill else "none",
                    edgecolors=COLORS[m], lw=1.2,
                )
                pts += 1
        # dense seed-0 baseline (Part A bench seed-0, dense path)
        ax.scatter([dense_lat[arch]], [dense_test[arch]], marker="*", s=160,
                   c="black", zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax.set_title(arch.upper())
        ax.set_xlabel(LAT_X[arch])
        if idx == 0:
            ax.set_ylabel("post_finetune test accuracy [-]")
        ax.grid(alpha=0.3, which="both")
        n_pts_per_panel = pts
    # 16 arms (4 methods x 4 target sparsities) x 2 paths
    assert n_pts_per_panel >= 32, (
        f"latency_vs_accuracy: expected >= 32 points per panel, got {n_pts_per_panel}"
    )
    _shared_legend(
        fig,
        [_h(COLORS[m], m, marker=SHAPES[m], ls="", markerfacecolor=COLORS[m],
            markeredgecolor=COLORS[m], ms=8) for m in METHODS]
        + [
            _h("0.3", "hollow = dense_zeros path", marker="o", ls="", mfc="none", ms=8),
            _h("0.3", "filled = compact path", marker="o", ls="", mfc="0.3", ms=8),
            _h("black", "dense seed-0 (Part A bench)", marker="*", ls="", ms=10),
        ],
        title="marker shape = method", ncol=5, pad=0.14,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _tight_savefig(fig, FIGURES / "latency_vs_accuracy.png", dpi=150)
    captions.append(
        "fig latency_vs_accuracy.png: 3 panels (MLP, CNN, GCN). x: median forward "
        "latency of each runtime path, batch 256 for MLP/CNN, full graph for GCN (ms, "
        "log scale), from the per-path timing blocks of results/pilot_summary.csv "
        "(phase=pilot, stage=post_finetune, seed-0: {path}_batch_256_median_ms / "
        "{path}_full_graph_median_ms); y: post_finetune test accuracy (-). One hollow "
        "marker per arm for the dense_zeros path and one filled marker for the compact "
        "path; marker shape = method (o s1, s s2, ^ s3, D s4) — 32 points per panel. "
        "Black star: dense seed-0 (results/baselines.md, Part A bench seed-0). Source: "
        "results/pilot_summary.csv, results/baselines.md."
    )
    plt.close(fig)

    # ----------------------------------------------------------------- c
    # size_vs_sparsity: 3 panels, bytes on disk (log) vs achieved
    # eligible-weight sparsity (0..1); one series per method per artifact
    # type (dense_zeros = dense_bytes, gzip = gzip_bytes, compact =
    # compressed_bytes), color = method, linestyle = artifact type.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for idx, arch in enumerate(ARCHS):
        ax = axes[idx]
        lstyle = {"dense_zeros": "-", "gzip": "--", "compact": "-."}
        for m in METHODS:
            src = sorted(
                (r for r in pilot_ft
                 if r["architecture"] == arch and r["method"] == m),
                key=lambda r: r["achieved_sparsity"] or 0.0,
            )
            for name in ("dense_zeros", "gzip", "compact"):
                col = (
                    "dense_bytes" if name == "dense_zeros"
                    else "gzip_bytes" if name == "gzip"
                    else "compressed_bytes"
                )
                pairs = sorted(
                    (r["achieved_sparsity"], r[col])
                    for r in src
                    if r.get("achieved_sparsity") is not None and r.get(col)
                )
                for x, y in pairs:
                    assert 0.0 <= x <= 1.0, (
                        f"achieved_sparsity {x} out of [0,1] for {arch}/{m} ({col})"
                    )
                    assert y > 1000, (
                        f"bytes {y} for {arch}/{m} {col} is not a byte count (>1000 "
                        "expected); a sparsity fraction was likely plotted as bytes"
                    )
                if pairs:
                    xs, ys = zip(*pairs)
                    ax.plot(
                        xs, ys, marker="o", ms=3, lw=1.2,
                        linestyle=lstyle[name], color=COLORS[m],
                    )
        ax.axhline(dense_bytes[arch], color="black", lw=1.0)
        ax.set_yscale("log")
        ax.set_title(arch.upper())
        ax.set_xlabel(XLABEL[arch])
        if idx == 0:
            ax.set_ylabel("bytes on disk (log scale)")
        ax.grid(alpha=0.3, which="both")
    _shared_legend(
        fig,
        [_h(COLORS[m], m, lw=1.4) for m in METHODS]
        + [
            _h("0.3", "solid = dense_zeros (dense_bytes, zeros still stored)", lw=1.2),
            _h("0.3", "dashed = gzip-9 of the dense state_dict (gzip_bytes)", lw=1.2, ls="--"),
            _h("0.3", "dash-dot = compact artifact (compressed_bytes; S2 rebuilt dense / S1,S3 CSR / S4 packed N:M)", lw=1.2, ls="-."),
            _h("black", "black = dense on-disk bytes (Part A bench seed-0)", lw=1.0),
        ],
        title="color = method / line style = artifact type", ncol=3, pad=0.16,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    _tight_savefig(fig, FIGURES / "size_vs_sparsity.png", dpi=150)
    captions.append(
        "fig size_vs_sparsity.png: 3 panels (MLP, CNN, GCN). x: achieved sparsity of "
        "eligible weights (fraction zero, -); y: bytes on disk (log scale). One series "
        "per method (color = method) and artifact type (line style = artifact): solid = "
        "dense_zeros path (dense_bytes, zeros still stored), dashed = gzip-9 of the "
        "dense state_dict (gzip_bytes), dash-dot = compact artifact (compressed_bytes; "
        "S2 rebuilt dense / S1,S3 CSR / S4 packed N:M). Black line: dense on-disk bytes "
        "(results/baselines.md, Part A bench seed-0). All byte values > 1000 B "
        "(asserted). Source: results/pilot_summary.csv (phase=pilot, stage=post_finetune "
        "rows: dense_bytes / gzip_bytes / compressed_bytes) and results/baselines.md."
    )
    plt.close(fig)

    with open(FIGURES / "CAPTIONS.md", "w") as f:
        f.write("# Figure captions\n\n")
        for c in captions:
            f.write("- " + c + "\n")
    # Part C precision figures + captions (appended after the Part B section)
    precision_main()
    print("wrote figures + captions")


PRECISION_FORMATS = ("fp16", "bf16", "int8_dynamic", "int8_static", "int4_simulated")
FORMAT_LABEL = {
    "fp16": "fp16", "bf16": "bf16",
    "int8_dynamic": "int8 dyn (qnnpack)", "int8_static": "int8 static (qnnpack)",
    "int4_simulated": "int4 sim",
}


def precision_main() -> None:
    """Part C precision figures: 5 formats x 3 architectures.

    Reads phase="precision" rows from runs.jsonl and writes:
      figures/partC_precision_barbox.png    grouped bar + IQR (val_acc per arch/format)
      figures/partC_precision_latency.png   grouped bar (median latency per arch/format, log y)
      figures/partC_precision_frac.png      grouped bar (frac_macs_reduced_precision per arch/format)
      figures/CAPTIONS.md                   appended captions
    """
    recs = _loads()
    FIGURES.mkdir(parents=True, exist_ok=True)
    prec = [r for r in recs if r.get("part") == "C" and r.get("phase") == "precision" and r.get("status") == "ok"]
    if not prec:
        print("no phase=precision rows; skipping Part C figures")
        return
    # group by (arch, format)
    d = {(r["architecture"], r["method"]): r for r in prec}
    captions: list[str] = []

    def lat(r: dict) -> float | None:
        latk = LAT_KEY[r["architecture"]]
        v = (r.get("latency") or {}).get(latk) or {}
        m = v.get("median_ms")
        return float(m) if isinstance(m, (int, float)) and m > 0 else None

    # ---- barbox: val_acc per arch (x=arch), color=fmt, IQR bar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    n = len(PRECISION_FORMATS)
    for idx, arch in enumerate(ARCHS):
        ax = axes[idx]
        x = np.arange(n)
        accs, errs, labels, colors = [], [], [], []
        for j, fmt in enumerate(PRECISION_FORMATS):
            r = d.get((arch, fmt))
            if r is None or r.get("val_acc") is None:
                continue
            accs.append(r["val_acc"]); labels.append(FORMAT_LABEL[fmt]); colors.append(fmt)
            iqr = (r.get("latency") or {}).get(LAT_KEY[arch], {}).get("iqr_ms")
            errs.append(0.0)  # val_acc is a single point; no IQR on accuracy
        if accs:
            bars = ax.bar(x[:len(accs)], accs, 0.7, color=[plt.cm.tab10(i) for i in range(len(accs))])
            ax.set_xticks(x[:len(accs)])
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("validation accuracy [-]")
            ax.grid(alpha=0.3, axis="y")
        ax.set_title(arch.upper())
        ax.set_ylim(min(accs) - 0.02, max(accs) + 0.02 if accs else (0, 1))
    _shared_legend(
        fig,
        [_h(plt.cm.tab10(i), FORMAT_LABEL[fmt], lw=1.4) for i, fmt in enumerate(PRECISION_FORMATS)],
        title="format", ncol=5, pad=0.10,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _tight_savefig(fig, FIGURES / "partC_precision_valacc.png", dpi=150)
    captions.append(
        "fig partC_precision_valacc.png: 3 panels (MLP, CNN, GCN). x: reduced-precision format; "
        "y: validation accuracy (-). 5 formats: fp16, bf16, int8_dynamic (qnnpack), "
        "int8_static (qnnpack FX), int4_simulated. Source: results/runs.jsonl, part=C, "
        "phase=precision, status=ok (val_acc). One bar per format per architecture."
    )
    plt.close(fig)

    # ---- latency: grouped bar, median latency (log y)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, arch in enumerate(ARCHS):
        ax = axes[idx]
        x = np.arange(n)
        lats, labels = [], []
        for j, fmt in enumerate(PRECISION_FORMATS):
            r = d.get((arch, fmt))
            if r is None: continue
            lv = lat(r)
            if lv is None: continue
            lats.append(lv); labels.append(FORMAT_LABEL[fmt])
        if lats:
            ax.bar(x[:len(lats)], lats, 0.7, color=[plt.cm.tab10(i) for i in range(len(lats))])
            ax.set_xticks(x[:len(lats)])
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_yscale("log")
            ax.set_ylabel(LAT_X[arch].split(" (ms")[0])
            ax.grid(alpha=0.3, axis="y")
        ax.set_title(arch.upper())
    _shared_legend(
        fig,
        [_h(plt.cm.tab10(i), FORMAT_LABEL[fmt], lw=1.4) for i, fmt in enumerate(PRECISION_FORMATS)],
        title="format", ncol=5, pad=0.10,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _tight_savefig(fig, FIGURES / "partC_precision_latency.png", dpi=150)
    captions.append(
        "fig partC_precision_latency.png: 3 panels (MLP, CNN, GCN). x: format; y: median "
        "forward latency (log scale, ms). Batch 256 for MLP/CNN, full graph for GCN "
        "(frozen bench protocol). Source: results/runs.jsonl, part=C, phase=precision, "
        "status=ok (latency). One bar per format per architecture."
    )
    plt.close(fig)

    # ---- frac_macs: fraction of MACs in reduced precision
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, arch in enumerate(ARCHS):
        ax = axes[idx]
        x = np.arange(n)
        fracs, labels = [], []
        for j, fmt in enumerate(PRECISION_FORMATS):
            r = d.get((arch, fmt))
            if r is None: continue
            f = r.get("frac_macs_reduced_precision")
            if f is None: continue
            fracs.append(float(f)); labels.append(FORMAT_LABEL[fmt])
        if fracs:
            ax.bar(x[:len(fracs)], fracs, 0.7, color=[plt.cm.tab10(i) for i in range(len(fracs))])
            ax.set_xticks(x[:len(fracs)])
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("frac MACs reduced-precision [-]")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3, axis="y")
        ax.set_title(arch.upper())
    _shared_legend(
        fig,
        [_h(plt.cm.tab10(i), FORMAT_LABEL[fmt], lw=1.4) for i, fmt in enumerate(PRECISION_FORMATS)],
        title="format", ncol=5, pad=0.10,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _tight_savefig(fig, FIGURES / "partC_precision_frac_macs.png", dpi=150)
    captions.append(
        "fig partC_precision_frac_macs.png: 3 panels (MLP, CNN, GCN). x: format; y: fraction "
        "of total forward MACs performed in reduced precision (-). For half formats all ops "
        "are reduced (1.0); for INT8 the fraction is the qnnpack-quantized ops (1.0 for MLP, "
        "1.0 for GCN projections, 0.5 for CNN where Conv2d stays fp32); INT4 simulated is 0.0 "
        "(fp32 dequant inside timed region). Source: results/runs.jsonl, part=C, phase=precision, "
        "status=ok (frac_macs_reduced_precision, per_layer)."
    )
    plt.close(fig)

    # Append captions to existing CAPTIONS.md
    cap_path = FIGURES / "CAPTIONS.md"
    header = "# Figure captions\n\n" if not cap_path.exists() else "\n## Part C precision\n\n"
    with open(cap_path, "a") as f:
        f.write(header)
        for c in captions:
            f.write("- " + c + "\n")
    print("wrote Part C precision figures + captions")


# (main() invocation moved to the single __main__ block at the bottom)


# ---------------------------------------------------------------------------
# C2: precision_size_latency + factorial figures
# ---------------------------------------------------------------------------

def _c2_rows(recs: list[dict], phase: str) -> dict[tuple[str, str], dict]:
    """Latest C2-run ok row per (architecture, method).

    C2 reran the entire Part C study on the pinned Python 3.12 environment. The
    C1b rows (Python 3.13) were superseded; C2 rows are the ones whose `code`
    hash and python version match the C2 run. We select the LATEST (file order)
    ok row per (arch, method) whose python is 3.12.x — the C2 environment.
    """
    out: dict[tuple[str, str], dict] = {}
    for r in recs:
        if r.get("part") != "C" or r.get("phase") != phase or r.get("status") != "ok":
            continue
        py = (r.get("versions") or {}).get("python", "")
        if not py.startswith("3.12"):
            continue  # skip C1b (3.13) superseded rows
        out[(r["architecture"], r["method"])] = r  # latest wins
    return out


def _readable_ticks(ax) -> None:
    """C5: make every axis carry at least 2 visible tick labels.

    Log axes get 1/2/5 major ticks with plain-number formatting (minor ticks
    off); linear axes keep their default locator (which yields >=2 labels).
    Used by the C2/C5 figures where narrow ranges made default log labels
    vanish.
    """
    import matplotlib.ticker as mtick
    for scale in ("x", "y"):
        axis = ax.xaxis if scale == "x" else ax.yaxis
        is_log = getattr(ax, f"get_{scale}scale", lambda: "linear")() == "log"
        getter = axis.get_major_locator
        setter = axis.set_major_locator
        if is_log:
            loc = mtick.LogLocator(subs=(1, 2, 5))
            setter(loc)
            if not isinstance(getter(), mtick.FuncFormatter):
                # plain-number formatting (e.g. 1.0e5 -> 100000)
                fmt = mtick.ScalarFormatter()
                fmt.set_scientific(False)
                fmt.set_useOffset(False)
                axis.set_major_formatter(fmt)
            ax.minorticks_off()
        # explicit assertion: >=2 visible major tick labels on this axis
        lo, hi = axis.get_view_interval()
        ticks = getter().tick_values(lo, hi)
        assert len(ticks) >= 2, (
            f"C5: {scale} axis has {len(ticks)} tick labels (<2): {ticks}")


def _replace_caption_section(path: Path, header: str, caption: str) -> None:
    """C5: upsert a '## <header>' section (replace all existing copies) so
    re-running the figure script does not duplicate captions."""
    ls = path.read_text().splitlines(keepends=True) if path.exists() else []
    kept, skip = [], False
    for l in ls:
        if l.startswith("## "):
            skip = l.rstrip("\n") == header.rstrip("\n")
            if skip:
                continue
        elif skip:
            continue
        kept.append(l)
    text = "".join(kept)
    if not text.endswith("\n"):
        text += "\n"
    lines = text.splitlines(keepends=True)
    inserted = f"\n{header.rstrip('\\n')}\n\n- {caption}\n\n"
    idx = 0
    for pos, l in enumerate(lines):
        if l.startswith("## "):
            idx = pos
            break
    else:
        idx = len(lines)
    lines.insert(idx, inserted)
    path.write_text("".join(lines).replace("\n\n\n\n", "\n\n"))


def precision_size_latency() -> None:
    """C2 item 8a: figures/precision_size_latency.png. Per arch: artifact bytes
    vs batch-256 / full-graph median latency, one point per format (plus the
    dense seed-0 FP32 reference from the latest factorial dense_fp32 row),
    readable plain-number ticks, unsupported formats in the caption."""
    recs = _loads()
    FIGURES.mkdir(parents=True, exist_ok=True)
    prec = _c2_rows(recs, "precision")
    if not prec:
        print("no C2 phase=precision rows; skipping precision_size_latency")
        return

    def lat(r: dict) -> float | None:
        v = (r.get("latency") or {}).get(LAT_KEY[r["architecture"]]) or {}
        m = v.get("median_ms")
        return float(m) if isinstance(m, (int, float)) and m > 0 else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    captions: list[str] = []
    for idx, arch in enumerate(ARCHS):
        ax = axes[idx]
        xs, ys, labels, colors = [], [], [], []
        for j, fmt in enumerate(PRECISION_FORMATS):
            r = prec.get((arch, fmt))
            if r is None:
                continue
            ab = r.get("artifact_bytes")
            lv = lat(r)
            if not ab or lv is None:
                continue
            xs.append(ab); ys.append(lv)
            labels.append(FORMAT_LABEL[fmt]); colors.append(fmt)
        # C5: dense seed-0 FP32 reference (latest factorial dense_fp32 row).
        fp32 = None
        for r in recs:
            if (r.get("part") == "C" and r.get("phase") == "factorial"
                    and r.get("status") == "ok" and r.get("architecture") == arch
                    and r.get("method") == "dense_fp32"
                    and (r.get("versions") or {}).get("python", "").startswith("3.12")):
                fp32 = r
        ref_x = ref_y = None
        if fp32 is not None:
            ref_x = fp32.get("artifact_bytes")
            ref_y = lat(fp32)
            if ref_x and ref_y is not None:
                xs.append(ref_x); ys.append(ref_y)
                labels.append("FP32 ref (dense)"); colors.append("fp32_ref")
        if xs:
            n = len(xs)
            sizes = [60] * (n - 1) + [140] if ref_x else [60] * n
            ax.scatter(xs, ys, s=sizes, color=[plt.cm.tab10(i % 10) for i in range(n)],
                       marker="o", zorder=3, edgecolors="k", linewidths=0.5)
            if ref_x:  # C5: FP32 reference point as a distinct star
                ax.scatter([ref_x], [ref_y], s=160, marker="*", c="black",
                           zorder=4, edgecolors="k", linewidths=0.5)
            for xi, yi, lb in zip(xs, ys, labels):
                ax.annotate(f"{lb}\n({xi:.4g} B, {yi:.3g} ms)",
                            (xi, yi), textcoords="offset points",
                            xytext=(5, 3), fontsize=6.5)
            allx, ally = xs, ys  # ref point already included in xs/ys
            for scale, vals in (("x", allx), ("y", ally)):
                lo, hi = min(vals), max(vals)
                if hi / lo < 10:  # under 1 decade: linear axis is more readable
                    getattr(ax, f"set_{scale}scale")("linear")
                else:
                    getattr(ax, f"set_{scale}scale")("log")
                    pad = max((hi - lo), 1e-9) * 0.05
                    getattr(ax, f"set_{scale}lim")(lo / 1.06, hi * 1.06)
            ax.set_xlabel("artifact bytes")
            ax.set_ylabel(f"median latency (ms, {LAT_KEY[arch]})")
            ax.grid(alpha=0.3, which="both")
            _readable_ticks(ax)
        else:
            ax.axis("off")
        ax.set_title(arch.upper())
    # unsupported formats (not present for some arch) -> caption
    missing = []
    for arch in ARCHS:
        for fmt in PRECISION_FORMATS:
            if (arch, fmt) not in prec:
                missing.append(f"{arch}/{fmt}")
    _shared_legend(
        fig,
        [_h(plt.cm.tab10(i), FORMAT_LABEL[fmt]) for i, fmt in enumerate(PRECISION_FORMATS)],
        title="format", ncol=5, pad=0.10,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _tight_savefig(fig, FIGURES / "precision_size_latency.png", dpi=150)
    cap = ("fig precision_size_latency.png: 3 panels (MLP, CNN, GCN). x: artifact bytes; "
           "y: median forward latency (ms; batch 256 for MLP/CNN, full graph for GCN). "
           "One point per format, each labeled with (bytes, ms); star = dense seed-0 "
           "FP32 reference (latest C2 factorial dense_fp32 row). Axes use plain-number "
           "1/2/5 log ticks when the range is >= 1 decade, linear otherwise. "
           "Source: results/runs.jsonl, part=C, phase=precision, status=ok, python 3.12 "
           "(C2 run); reference from phase=factorial. "
           + ("Unsupported/absent formats (not plotted): " + ", ".join(missing) + "."
              if missing else "All 5 formats present for all architectures."))
    captions.append(cap)
    plt.close(fig)

    _replace_caption_section(FIGURES / "CAPTIONS.md", "## Part C (C2) size-latency", cap)
    print("wrote precision_size_latency.png + caption")


def factorial_fig() -> None:
    """C2 item 8b: figures/factorial.png. Per arch: accuracy, bytes, latency
    for the 4 conditions (dense_fp32, sparse_fp32, dense_int8, sparse_int8).
    Three sub-panels per arch (acc / bytes / latency), one point per condition."""
    recs = _loads()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fact: dict[tuple[str, str], dict] = {}
    for r in recs:
        if (r.get("part") != "C" or r.get("phase") != "factorial"
                or r.get("status") != "ok"):
            continue
        py = (r.get("versions") or {}).get("python", "")
        if not py.startswith("3.12"):
            continue
        fact[(r["architecture"], r["method"])] = r
    if not fact:
        print("no C2 phase=factorial rows; skipping factorial.png")
        return

    conds = ("dense_fp32", "sparse_fp32", "dense_int8", "sparse_int8")
    cond_label = {"dense_fp32": "dense fp32", "sparse_fp32": "sparse fp32",
                  "dense_int8": "dense int8", "sparse_int8": "sparse int8"}

    def lat(r: dict) -> float | None:
        v = (r.get("latency") or {}).get(LAT_KEY[r["architecture"]]) or {}
        m = v.get("median_ms")
        return float(m) if isinstance(m, (int, float)) and m > 0 else None

    panels = [("val_acc", "validation accuracy [-]", False),
              ("artifact_bytes", "artifact bytes", True),
              ("lat", "median latency (ms)", True)]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for pidx, (pkey, ylabel, logy) in enumerate(panels):
        for idx, arch in enumerate(ARCHS):
            ax = axes[pidx][idx]
            xs, ys, labels = [], [], []
            for j, cond in enumerate(conds):
                r = fact.get((arch, cond))
                if r is None:
                    continue
                if pkey == "val_acc":
                    val = r.get("val_acc")
                elif pkey == "artifact_bytes":
                    val = r.get("artifact_bytes")
                else:
                    val = lat(r)
                if val is None:
                    continue
                xs.append(j); ys.append(float(val)); labels.append(cond_label[cond])
            if xs:
                ax.bar(xs, ys, 0.6, color=[plt.cm.tab10(i) for i in range(len(xs))])
                ax.set_xticks(xs)
                ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
                if logy:
                    ax.set_yscale("log")
                # C5: value on top of every bar (bytes in KB, latency in ms with
                # 3 sig digits, accuracy with 4 decimals).
                for xi, yi in zip(xs, ys):
                    if pkey == "val_acc":
                        txt = f"{yi:.4f}"
                    elif pkey == "artifact_bytes":
                        txt = f"{yi / 1024:.4g} KB"
                    else:
                        txt = f"{yi:.3g} ms"
                    ax.annotate(txt, (xi, yi), textcoords="offset points",
                                xytext=(0, 3), ha="center", fontsize=6.5)
                if pkey == "val_acc":
                    ax.set_ylim(min(ys) - 0.02, max(ys) + 0.015)
                else:
                    ax.set_ylim(min(ys) / 1.1, max(ys) * 1.8)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.grid(alpha=0.3, axis="y")
                # C5: readable tick labels on the value axis.
                if logy:
                    import matplotlib.ticker as mtick
                    ax.yaxis.set_major_locator(mtick.LogLocator(subs=(1, 2, 5)))
                    if pkey == "artifact_bytes":
                        ax.yaxis.set_major_formatter(
                            mtick.FuncFormatter(lambda v, _: f"{v / 1024:.3g}"))
                        ax.set_ylabel("artifact bytes (KB)", fontsize=9)
                    else:
                        ax.yaxis.set_major_formatter(
                            mtick.FuncFormatter(lambda v, _: f"{v:.3g}"))
                    ax.minorticks_off()
                _readable_ticks(ax)
            ax.set_title(arch.upper(), fontsize=10)
    # C5: x tick labels already name the conditions -> drop the shared legend
    # (it left a large empty band above the grid).
    fig.tight_layout()
    _tight_savefig(fig, FIGURES / "factorial.png", dpi=150)
    cap = ("fig factorial.png: 3x3 grid (rows = metric: validation accuracy, artifact "
           "bytes, median latency; columns = MLP/CNN/GCN). One bar per condition: "
           "dense_fp32, sparse_fp32, dense_int8, sparse_int8 (the C2-selected INT8 "
           "method applied to dense and to the sparse operating point); each bar is "
           "labeled with its value (accuracy to 4 decimals, bytes in KB, latency in ms "
           "with 3 significant digits). Bytes and latency on log scale with plain-number "
           "ticks; no shared legend (x tick labels name the conditions). "
           "Source: results/runs.jsonl, part=C, phase=factorial, "
           "status=ok, python 3.12 (C2 run).")
    _replace_caption_section(FIGURES / "CAPTIONS.md", "## Part C (C2) factorial", cap)
    print("wrote factorial.png + caption")


if __name__ == "__main__":
    import sys as _s
    if len([a for a in _s.argv[1:] if not a.startswith("-")]) == 0:
        main()
    if "size-latency" in _s.argv:
        precision_size_latency()
    if "factorial" in _s.argv:
        factorial_fig()
