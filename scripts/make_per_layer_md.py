"""Task B3 item 4: results/per_layer_sparsity.md.

For every pilot arm (stage=post_finetune) a per-layer table (layer, shape,
numel, nonzero, sparsity), grouped by architecture and method. S2 tables use
the per_layer entries recorded on the arm (the physically REBUILT model's
shapes). Generated deterministically from results/runs.jsonl phase=pilot rows
(re-runnable); does not modify the log.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "code"))
from lab3.prune import per_layer_table  # noqa: E402

RESULTS = _REPO / "results"
RUNS = RESULTS / "runs.jsonl"
ARCHS = ("mlp", "cnn", "gcn")
METHODS = ("s1", "s2", "s3", "s4")


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


def _dense_parent_shapes(arch: str) -> list[tuple[str, str]]:
    """Current per_layer (name, 'o x i ...') for the dense architecture.

    Rebuilt from the dense PARENT's shapes at the time this file is generated
    (used only to label how far the S2 shapes shrink; the S2 tables themselves
    carry the recorded rebuilt shapes from the pilot rows)."""
    from lab3.models import build_model

    m = build_model(arch)
    out = []
    for e in per_layer_table(m):
        out.append((e["name"], "x".join(map(str, e["shape"]))))
    return out


def main() -> None:
    recs = _loads()
    rows = [
        r
        for r in recs
        if r.get("part") == "B"
        and r.get("phase") == "pilot"
        and r.get("stage") == "post_finetune"
        and r.get("status") == "ok"
    ]
    by_arm: dict[tuple, dict] = {}
    for r in rows:
        by_arm[(r["architecture"], r["method"], r["target_sparsity"])] = r

    dense_shapes = {a: _dense_parent_shapes(a) for a in ARCHS}

    lines: list[str] = []
    lines.append("# Per-layer sparsity — Part B pilot, post_finetune (seed 0)\n")
    lines.append(
        "One table per pilot arm (phase=pilot, stage=post_finetune, status=ok), "
        "grouped by architecture and method. Columns: layer (state key of the "
        "eligible weight tensor), shape (the layer's actual weight shape in the "
        "post-fine-tune checkpoint — for S2 this is the physically REBUILT "
        "model's smaller shape, recorded on the arm row), numel, nonzero, "
        "sparsity = 1 - nonzero/numel. Achieved sparsity (eligible-weight "
        "fraction zero, relative to the dense parent) is shown per arm. "
        "Source: results/runs.jsonl (phase=pilot rows, `per_layer` field); "
        "dense parent shapes below each method section are shown for "
        "reference.\n"
    )
    for arch in ARCHS:
        lines.append(f"## {arch.upper()}\n")
        lines.append(
            "Dense parent shapes: "
            + ", ".join(f"{n} {s}" for n, s in dense_shapes[arch])
            + "\n"
        )
        for m in METHODS:
            lines.append(f"### {m.upper()}\n")
            targets = sorted(
                (
                    r
                    for (a, mm, t), r in by_arm.items()
                    if a == arch and mm == m
                ),
                key=lambda r: r["target_sparsity"],
            )
            if not targets:
                lines.append("_no ok pilot rows_\n")
                continue
            for r in targets:
                t = r["target_sparsity"]
                pat = ""
                if m == "s4":
                    from lab3.sparsity import nm_pattern_for_target

                    n, mm_ = nm_pattern_for_target(t)
                    pat = f" (pattern {n}:{mm_})"
                if m == "s2":
                    gs = (r.get("method_params") or {}).get("group_sparsity_achieved")
                    pat = f" (group sparsity achieved {gs:.4f})" if gs is not None else ""
                lines.append(
                    f"#### target {t:.2f}{pat} — achieved eligible sparsity "
                    f"{r.get('achieved_sparsity'):.4f}, arm {r.get('arm_id')}\n"
                )
                lines.append(
                    "| layer | shape | numel | nonzero | sparsity |\n"
                    "|---|---|---|---|---|"
                )
                for e in r.get("per_layer") or []:
                    shape = "x".join(map(str, e["shape"]))
                    lines.append(
                        f"| {e['name']} | {shape} | {e['numel']} | "
                        f"{e['nonzero']} | {e['sparsity']:.4f} |"
                    )
                # note removed tensors for S2 (rebuilt shapes differ from parent)
                if m == "s2":
                    rec_names = {e["name"] for e in r.get("per_layer") or []}
                    parent_names = {n for n, _ in dense_shapes[arch]}
                    removed = sorted(parent_names - rec_names)
                    lines.append("")
                    if removed:
                        lines.append(
                            f"_rebuilt model omits: {', '.join(removed)}; "
                            "shown shapes are the rebuilt model's shapes, "
                            "not the dense parent's._"
                        )
                    lines.append("")
    out = RESULTS / "per_layer_sparsity.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(rows)} pilot arms)")


if __name__ == "__main__":
    main()
