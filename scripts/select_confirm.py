"""Apply the frozen predictions.md B.3 rule (validation only) to the seed-0 pilot.
Reads ONLY val_acc; test_acc is never loaded into the decision."""
import json, sys
DENSE_VAL = {}
TAU = {"mlp": 0.010, "cnn": 0.010, "gcn": 0.020}
NVAL = {"mlp": 5000, "cnn": 5000, "gcn": 500}
# Compare integer counts of correct validation examples: val_acc is float32, and
# 390/500 = 0.78 is stored as 0.77999997, which a float >= would wrongly reject.
cnt = lambda a, acc: round(acc * NVAL[a])
rows = [json.loads(l) for l in open("results/runs.jsonl") if l.strip()]
for r in rows:
    if r.get("part") == "A" and r.get("command") == "train" and r.get("seed") == 0 and r.get("status") == "ok":
        DENSE_VAL[r["architecture"]] = r["val_acc"]
pilot = [r for r in rows if r.get("phase") == "pilot" and r.get("stage") == "post_finetune"
         and r.get("status") == "ok" and r.get("method") in ("s1", "s2", "s3", "s4")]
out = []
for arch in ("mlp", "cnn", "gcn"):
    for m in ("s1", "s2", "s3", "s4"):
        cells = sorted([(r["target_sparsity"], r["val_acc"], r["run_id"]) for r in pilot
                        if r["architecture"] == arch and r["method"] == m])
        thr = DENSE_VAL[arch] - TAU[arch]
        thr_cnt = cnt(arch, DENSE_VAL[arch]) - round(TAU[arch] * NVAL[arch])
        elig = [c for c in cells if cnt(arch, c[1]) >= thr_cnt]
        if elig:
            pick, why = max(elig, key=lambda c: c[0]), "highest eligible sparsity"
        else:
            pick = sorted(cells, key=lambda c: (-c[1], c[0]))[0]; why = "fallback: best val"
        out.append({"arch": arch, "method": m, "dense_val": DENSE_VAL[arch], "threshold": round(thr, 4), "threshold_correct_count": thr_cnt, "n_val": NVAL[arch],
                    "pilot_val_by_target": {f"{c[0]:.2f}": c[1] for c in cells},
                    "selected_target": pick[0], "selected_run_id": pick[2], "rule": why})
json.dump(out, open("results/confirm_selection.json", "w"), indent=1)
for o in out:
    print(f"{o['arch']} {o['method']} thr={o['threshold']:.4f} vals={o['pilot_val_by_target']} -> {o['selected_target']} ({o['rule']})")
