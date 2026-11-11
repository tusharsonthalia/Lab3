"""Apply the frozen predictions.md C.1 sparse-operating-point rule. Eligibility uses only val_acc
(mean over confirm seeds 1-3, compared in correct-example counts); the choice among eligible points
uses measured compact-artifact bytes of the seed-0 pilot checkpoint (the factorial parent).
test_acc is never read."""
import json
rows = [json.loads(l) for l in open("results/runs.jsonl") if l.strip()]
sel = json.load(open("results/confirm_selection.json"))
TAU = {"mlp": 0.010, "cnn": 0.010, "gcn": 0.020}; NVAL = {"mlp": 5000, "cnn": 5000, "gcn": 500}
cnt = lambda a, x: round(x * NVAL[a])
dense_val = {r["architecture"]: r["val_acc"] for r in rows
             if r.get("part") == "A" and r.get("command") == "train" and r.get("seed") == 0 and r["status"] == "ok"}
out = {}
for arch in ("mlp", "cnn", "gcn"):
    thr = cnt(arch, dense_val[arch]) - round(TAU[arch] * NVAL[arch])
    cands = []
    for s in sel:
        if s["arch"] != arch: continue
        m, t = s["method"], s["selected_target"]
        conf = [r for r in rows if r.get("phase") == "confirm" and r.get("stage") == "post_finetune" and r["status"] == "ok"
                and r["architecture"] == arch and r["method"] == m and abs(r["target_sparsity"] - t) < 1e-9]
        pilot = [r for r in rows if r.get("phase") == "pilot" and r.get("stage") == "post_finetune" and r["status"] == "ok"
                 and r["architecture"] == arch and r["method"] == m and abs(r["target_sparsity"] - t) < 1e-9]
        assert len(conf) == 3 and len(pilot) == 1, (arch, m, len(conf), len(pilot))
        mean_cnt = sum(cnt(arch, r["val_acc"]) for r in conf) / 3
        cands.append({"method": m, "target": t, "mean_val_count": mean_cnt, "mean_val": mean_cnt / NVAL[arch],
                      "eligible": mean_cnt >= thr, "compact_bytes_seed0": pilot[0]["compressed_bytes"],
                      "pilot_run_id": pilot[0]["run_id"], "pilot_arm_id": pilot[0]["arm_id"]})
    elig = [c for c in cands if c["eligible"]]
    if elig:
        best = min(c["compact_bytes_seed0"] for c in elig)
        tied = [c for c in elig if c["compact_bytes_seed0"] <= best * 1.01]
        pick = max(tied, key=lambda c: c["mean_val_count"]); why = f"smallest compact bytes; {len(tied)} within 1% -> higher mean val"
    else:
        pick = max(cands, key=lambda c: c["mean_val_count"]); why = "fallback: highest mean val"
    out[arch] = {"threshold_count": thr, "n_val": NVAL[arch], "candidates": cands, "selected": pick, "rule": why}
    print(arch, "thr", thr, [(c["method"], c["target"], round(c["mean_val"], 4), c["eligible"], c["compact_bytes_seed0"]) for c in cands],
          "->", pick["method"], pick["target"], "|", why)
json.dump(out, open("results/partc_selection.json", "w"), indent=1)
