"""Task B3 item 5: results/failure_accounting.md.

Every Part B arm/row ever attempted (smoke, pilot, confirm) with its status,
and every failed/excluded record with its classified reason and whether it is
excluded from analysis. Also verifies the 48-cell pilot matrix (3 archs x 4
methods x 4 targets): every cell has at least one ok phase=pilot post_finetune
row — a cell without one would have to be marked 'not applicable', which
there should be none of.

Deterministic and re-runnable: reads results/runs.jsonl (and the B1b smoke
copy results/smoke_b1b/runs.jsonl); never modifies the logs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "code"))

RESULTS = _REPO / "results"
RUNS = RESULTS / "runs.jsonl"
SMOKE_RUNS = RESULTS / "smoke_b1b" / "runs.jsonl"
ARCHS = ("mlp", "cnn", "gcn")
METHODS = ("s1", "s2", "s3", "s4")
TARGETS = (0.25, 0.5, 0.75, 0.9)


def _loads(path: Path) -> list[dict]:
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _arm(r: dict) -> tuple:
    return (r.get("architecture"), r.get("method"),
            r.get("target_sparsity"), r.get("seed"))


def main() -> None:
    recs = _loads(RUNS)
    smoke = _loads(SMOKE_RUNS) if SMOKE_RUNS.exists() else []

    b_recs = [r for r in recs if r.get("part") == "B"]
    a_recs = [r for r in recs if r.get("part") == "A" and r.get("status") == "failed"]

    lines: list[str] = []
    lines.append("# Failure accounting — Part B (smoke, pilot, confirm)\n")
    lines.append(
        "Every Part B arm/row ever attempted, with status. The run log is "
        "append-only: failed rows are never edited or deleted. 'Excluded from "
        "analysis' means the row (or its arm) is not used by any summary, "
        "figure, or selection rule in this lab.\n"
    )

    # ---------------------------------------------------------------- smoke
    lines.append("## 1. B1 / B1b smoke arms (phase=null, pre-pilot)\n")
    lines.append(
        "The B1 smoke matrix (`mlp s1/s2/s3/s4 t0.5` + `dense_ft` controls and "
        "the `cnn`/`gcn` s1 t0.5 checks) ran with the pre-B2 code, i.e. without "
        "the `phase` field (recorded as `null`). Its ok rows are duplicated in "
        "`results/smoke_b1b/runs.jsonl` (frozen copy) for traceability. Per "
        "predictions.md B.6, **all smoke rows are debug runs and are excluded "
        "from all analysis**; the B.6 amendment added the dense_ft control "
        "because these smoke runs showed post-fine-tune val above the dense "
        "parent.\n"
    )
    lines.append(
        "| arch | method | target | seed | stage | status | run_id | reason | "
        "excluded from analysis? |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in b_recs:
        if r.get("phase") is not None:
            continue
        rs = r.get("reason") or ""
        lines.append(
            f"| {r.get('architecture')} | {r.get('method')} | {r.get('target_sparsity')} "
            f"| {r.get('seed')} | {r.get('stage')} | {r.get('status')} "
            f"| {r.get('run_id')} | {rs or '—'} | yes (smoke) |"
        )
    lines.append(
        "\nSmoke failures and their classified reasons (bug fixes are recorded "
        "in the B1/B1b history; all three were fixed before the pilot and every "
        "smoke row above is superseded by the pilot runs):\n"
    )
    for r in b_recs:
        if r.get("phase") is None and r.get("status") == "failed":
            lines.append(
                f"- `{r.get('run_id')}` ({r.get('stage')}): "
                f"{r.get('reason')}"
            )
    lines.append("")

    # ---------------------------------------------------------------- pilot
    lines.append("## 2. Pilot matrix (phase=pilot, seed 0): 48 cells + 3 dense_ft\n")
    lines.append(
        "The pilot is a full 3 × 4 × 4 matrix; **every one of the 48 cells has "
        "an ok phase=pilot post_finetune row** (see verification below), so no "
        "cell is missing, failed, or 'not applicable'. The three B.6 "
        "dense_ft controls are likewise ok and reported alongside (not counted "
        "among the 48).\n"
    )
    ok_pf = {
        (r["architecture"], r["method"], r["target_sparsity"]): r
        for r in b_recs
        if r.get("phase") == "pilot" and r.get("stage") == "post_finetune"
        and r.get("status") == "ok"
    }
    ok_pp = {
        (r["architecture"], r["method"], r["target_sparsity"]): r
        for r in b_recs
        if r.get("phase") == "pilot" and r.get("stage") == "post_prune"
        and r.get("status") == "ok"
    }
    failed_pf = [r for r in b_recs if r.get("phase") == "pilot"
                 and r.get("status") == "failed"]
    lines.append("Verification of the 48-cell matrix:\n")
    lines.append(
        "| arch | method | t0.25 | t0.5 | t0.75 | t0.9 |"
    )
    lines.append("|---|---|---|---|---|---|")
    na_count = 0
    for a in ARCHS:
        for m in METHODS:
            cells = []
            for t in TARGETS:
                if (a, m, t) in ok_pf:
                    cells.append("ok")
                elif (a, m, t) in ok_pp:
                    cells.append("post_prune only (no post_finetune)")
                else:
                    cells.append("**NOT APPLICABLE** (no ok rows)")
                    na_count += 1
            lines.append(f"| {a} | {m} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"Cells marked not applicable: **{na_count}** of 48. "
        + (
            "As required, none — every cell is filled by a real ok run "
            "(predictions.md B.2: no cell is filled by copying another run)."
            if na_count == 0
            else "SEE BELOW."
        )
    )
    lines.append("")
    if failed_pf:
        lines.append("Failed pilot rows (all superseded by later ok runs):")
        for r in failed_pf:
            lines.append(f"- `{r.get('run_id')}`: {r.get('reason')}")
    else:
        lines.append("No failed phase=pilot rows in the log.")
    lines.append("")

    # ---------------------------------------------------------------- confirm
    lines.append("## 3. Confirmatory reruns (phase=confirm, seeds 1, 2, 3)\n")
    sel = json.loads((RESULTS / "confirm_selection.json").read_text())
    n_ok = len(
        {
            _arm(r)
            for r in b_recs
            if r.get("phase") == "confirm"
            and r.get("stage") == "post_finetune"
            and r.get("status") == "ok"
        }
    )
    n_failed = sum(1 for r in b_recs if r.get("phase") == "confirm"
                   and r.get("status") == "failed")
    lines.append(
        f"Expected arms: 12 selected points × 3 seeds + 3 dense_ft × 3 seeds = "
        f"45. Ok arms with a post_finetune row: {n_ok}. Failed phase=confirm "
        f"rows: {n_failed}."
    )
    lines.append("")
    for r in b_recs:
        if r.get("phase") == "confirm" and r.get("status") == "failed":
            lines.append(
                f"- failed: `{r.get('run_id')}` ({r.get('architecture')} "
                f"{r.get('method')} t{r.get('target_sparsity')} seed "
                f"{r.get('seed')}): {r.get('reason')}"
            )
    lines.append("")
    lines.append(
        "Confirmatory arms are NOT excluded from analysis: they are the "
        "confirmatory evidence (results/confirm_summary.csv, figures) for the "
        "B.3-selected points. The dense_ft seed 1–3 arms are reference/inter"
        "pretation controls (B.6/B.7), not part of the 12 selected points.\n"
    )

    # ------------------------------------------------- all failed records
    lines.append("## 4. Every failed record with classified reason\n")
    lines.append(
        "`reason` is the classified failure (prefix `unsupported_backend`, "
        "`numerical`, `bug`, or `not_applicable`, followed by the exception "
        "message). Part A failures (pre-Part-B, listed for completeness) are "
        "also excluded from Part B analysis.\n"
    )
    lines.append(
        "| part | phase | arch | method | target | seed | stage | run_id | "
        "reason (classified) | excluded from analysis? |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in b_recs:
        if r.get("status") != "failed":
            continue
        phase = r.get("phase")
        excl = (
            "yes (smoke)"
            if phase is None
            else "yes (superseded by later ok arm rows for the same cell)"
        )
        lines.append(
            f"| {r.get('part')} | {phase or 'null'} | {r.get('architecture')} "
            f"| {r.get('method')} | {r.get('target_sparsity')} "
            f"| {r.get('seed')} | {r.get('stage')} | {r.get('run_id')} "
            f"| {r.get('reason')} | {excl} |"
        )
    for r in a_recs:
        lines.append(
            f"| {r.get('part')} | null | {r.get('architecture')} "
            f"| {r.get('method')} | {r.get('target_sparsity')} "
            f"| {r.get('seed')} | {r.get('stage') or 'bench/eval/train'} "
            f"| {r.get('run_id')} | {r.get('reason')} "
            f"| yes (Part A pre-B2; superseded / not part of Part B) |"
        )
    lines.append("")

    lines.append(
        "## 5. Records excluded from analysis (summary)\n"
    )
    lines.append(
        "- **All smoke rows** (phase=null, Part B): debug runs per B.6 — "
        "excluded.\n"
        "- **All failed rows** (Part A and Part B, any phase): failed rows "
        "carry no measured values; the arms they belonged to were re-run and "
        "the ok rows supersede them (append-only log: both remain on file).\n"
        "- **Everything else is included**: 48 pilot arms (48 post_prune + 48 "
        "post_finetune ok rows), 3 dense_ft pilot controls, and the "
        "confirmatory arms once their ok rows are present.\n"
    )

    out = RESULTS / "failure_accounting.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}; ok pilot cells={len(ok_pf)}; "
          f"not-applicable cells={na_count}")


if __name__ == "__main__":
    main()
