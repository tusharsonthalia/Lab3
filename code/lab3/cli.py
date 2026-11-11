"""CLI for the Lab 3 dense-reference package.

Usage: uv run python -m lab3 <cmd> --model {mlp,cnn,gcn} ...
Commands: make-splits, train, eval, bench.
Configs live in configs/<model>.yaml and hold the frozen hyperparameters.
CLI flags may override config values for tests only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import torch

from .util import (
    CONFIGS_DIR,
    DATA_DIR,
    SPLIT_SEED,
    SPLITS_DIR,
    compute_splits_hash,
    set_threads,
)


def load_config(architecture: str) -> dict:
    path = CONFIGS_DIR / f"{architecture}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing config {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _cora_masks() -> tuple:
    from torch_geometric.datasets import Planetoid
    from .util import DATA_DIR as _D
    ds = Planetoid(root=str(_D / "Planetoid"), name="Cora")
    g = ds[0]
    return g.train_mask, g.val_mask, g.test_mask


def make_splits_impl() -> None:
    set_threads()
    import lab3.util as U
    existing = [p for p in U.SPLIT_FILES["fmnist"] + U.SPLIT_FILES["cora"] + [U.SPLITS_SHA]]
    if any(p.exists() for p in existing):
        raise FileExistsError("splits already exist; refusing to regenerate. "
                              "Delete splits/ manually only if you intend to reset all arms.")
    U.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SPLIT_SEED)

    from .data import _fmnist_dataset
    train_ds = _fmnist_dataset("train")
    labels = np.asarray(train_ds.targets)
    assert labels.shape[0] == 60000
    val_idx = []
    for cls in range(10):
        pool = np.flatnonzero(labels == cls)
        val_idx.append(np.random.permutation(pool)[:500])
    val_idx = np.sort(np.concatenate(val_idx))
    train_idx = np.setdiff1d(np.arange(60000), val_idx, assume_unique=True)
    test_idx = np.arange(10000)
    assert train_idx.shape[0] == 55000 and val_idx.shape[0] == 5000

    np.save(U.SPLITS_DIR / "fmnist_train_idx.npy", train_idx.astype(np.int64))
    np.save(U.SPLITS_DIR / "fmnist_val_idx.npy", val_idx.astype(np.int64))
    np.save(U.SPLITS_DIR / "fmnist_test_idx.npy", test_idx.astype(np.int64))

    tr_m, va_m, te_m = _cora_masks()
    cora_tr = np.flatnonzero(tr_m.numpy())
    cora_va = np.flatnonzero(va_m.numpy())
    cora_te = np.flatnonzero(te_m.numpy())
    assert cora_tr.shape[0] == 140 and cora_va.shape[0] == 500 and cora_te.shape[0] == 1000
    np.save(U.SPLITS_DIR / "cora_train_idx.npy", cora_tr.astype(np.int64))
    np.save(U.SPLITS_DIR / "cora_val_idx.npy", cora_va.astype(np.int64))
    np.save(U.SPLITS_DIR / "cora_test_idx.npy", cora_te.astype(np.int64))

    U.SPLITS_SHA.write_text(compute_splits_hash() + "  splits/*.npy (fixed order, see compute_splits_hash)\n")
    print(f"splits written to {U.SPLITS_DIR}; hash={compute_splits_hash()}")


def _apply_overrides(cfg: dict, args) -> dict:
    """CLI flags override frozen config values (for tests only)."""
    for flag, key in (("--lr", "lr"), ("--weight-decay", "weight_decay"),
                      ("--batch-size", "batch_size"), ("--epochs", "max_epochs"),
                      ("--patience", "patience")):
        val = getattr(args, flag.replace("-", "_"), None)
        if val is not None:
            cfg[key] = val
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lab3")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-splits")
    p.set_defaults(fn=make_splits_impl)

    def _make_calib():
        from .util import make_calib_split
        make_calib_split()

    p = sub.add_parser("make-calib", help="create the B.1 calibration subset once")
    p.set_defaults(fn=_make_calib)

    for name, help_ in (("train", "train a dense model"),
                        ("eval", "evaluate a saved checkpoint"),
                        ("bench", "benchmark a saved checkpoint"),
                        ("prune", "run one Part B sparsity arm")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--model", required=True, choices=["mlp", "cnn", "gcn"])
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--run-id", default=None)
        if name == "prune":
            p.add_argument("--no-bench", action="store_true",
                           help="skip the B.4 runtime paths (faster smoke)")
        # overrides (tests only)
        p.add_argument("--lr", type=float, default=None)
        p.add_argument("--weight-decay", type=float, default=None)
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--epochs", type=int, default=None)
        p.add_argument("--patience", type=int, default=None)

    # B2: pilot driver — runs the full frozen pilot matrix
    # {mlp,cnn,gcn} x {s1,s2,s3,s4} x {0.25,0.50,0.75,0.90} + dense_ft controls
    # at seed 0, phase="pilot" (predictions.md B.1/B.2/B.6).
    p = sub.add_parser("pilot", help="run the full Part B pilot matrix (seed 0)")
    p.add_argument("--arch", choices=["mlp", "cnn", "gcn"], default=None,
                   help="restrict to one architecture (default: all three)")
    p.add_argument("--only", default=None,
                   help="comma-separated (method,target) cells to run only these "
                        "(e.g. 's1,0.75' or 's1,0.75;s4,0.9'); dense_ft controls "
                        "are selected with 'dense_ft,0.0'")

    # B3: confirm driver — reruns the 12 selected points from
    # results/confirm_selection.json (frozen B.3/B.7 rule) with seeds 1, 2, 3
    # + the dense_ft control with seeds 1, 2, 3 per architecture; 45 arms,
    # phase="confirm" (predictions.md B.1/B.3/B.6/B.7).
    p = sub.add_parser("confirm",
                       help="run the B.3 confirmatory reruns (seeds 1, 2, 3)")
    p.add_argument("--arch", choices=["mlp", "cnn", "gcn"], default=None,
                   help="restrict to one architecture (default: all three)")
    p.add_argument("--only", default=None,
                   help="comma-separated (method,target) cells to run only these "
                        "(e.g. 's1,0.75' or 's1,0.75;s4,0.9'); dense_ft controls "
                        "are selected with 'dense_ft,0.0'")

    # C2: Part C (reduced precision) drivers, moved here from the standalone
    # code/lab3_precision.py module (now removed).
    p = sub.add_parser("precision",
                       help="Part C standalone precision study (5 formats x 3 archs)")
    p.add_argument("--arch", choices=["mlp", "cnn", "gcn"], default=None,
                   help="restrict to one architecture (default: all three)")

    p = sub.add_parser("factorial",
                       help="Part C 12-condition sparse x INT8 factorial + INT8 selection")
    p.add_argument("--arch", choices=["mlp", "cnn", "gcn"], default=None,
                   help="restrict to one architecture (default: all three)")

    p = sub.add_parser("c2summary",
                       help="write results/partc_summary.md from the C2 run rows")

    p = sub.add_parser("c4rerun",
                       help="C4: re-run the 12 INT8 rows (standalone + factorial) "
                            "with the flat state_dict artifact format")
    p.add_argument("--arch", choices=["mlp", "cnn", "gcn"], default=None,
                   help="restrict to one architecture (default: all three)")

    # prune-specific flags
    p = sub.choices["prune"]
    # B1b: `dense_ft` is the B.6 control arm (fine-tune the dense parent with
    # the B.1 budget, no pruning); it uses target 0.0. The s1-s4 methods keep
    # their frozen targets.
    p.add_argument("--method", required=True,
                   choices=["s1", "s2", "s3", "s4", "dense_ft"])
    p.add_argument("--target", type=float, required=True,
                   choices=[0.0, 0.25, 0.5, 0.75, 0.9])

    args = parser.parse_args(argv)
    set_threads()

    if args.cmd == "make-splits":
        args.fn()
        return 0
    if args.cmd == "make-calib":
        args.fn()
        return 0

    if args.cmd == "pilot":
        from .prune import run_pilot
        run_pilot(seed=0, arch=args.arch, only=args.only)
        return 0

    if args.cmd == "confirm":
        from .prune import run_confirm
        run_confirm(arch=args.arch, only=args.only)
        return 0

    if args.cmd == "precision":
        from . import c2 as c2
        c2.run_precision(arch=args.arch)
        return 0

    if args.cmd == "factorial":
        from . import c2 as c2
        c2.run_int8_selection_and_factorial(arch=args.arch)
        return 0

    if args.cmd == "c2summary":
        from . import c2 as c2
        c2.write_summary()
        return 0

    if args.cmd == "c4rerun":
        from . import c2 as c2
        c2.run_c4_artifact_rerun(arch=args.arch)
        return 0

    cfg = load_config(args.model)
    cfg = _apply_overrides(cfg, args)

    if args.cmd == "train":
        from .training import train
        rec = train(args.model, cfg, seed=args.seed, run_id=args.run_id)
        print(f"run_id={rec['run_id']} best_epoch={rec['best_epoch']} "
              f"val_acc={rec['val_acc']:.4f} test_acc={rec['test_acc']:.4f} "
              f"epochs={rec['train_epochs']} wall={rec['wall_time_s']}s")
    elif args.cmd == "eval":
        from .evaln import eval as do_eval
        rec = do_eval(args.model, seed=args.seed)
        print(f"run_id={rec['run_id']} val_acc={rec['val_acc']:.4f} test_acc={rec['test_acc']:.4f}")
    elif args.cmd == "bench":
        from .bench import bench
        rec = bench(args.model, seed=args.seed)
        lat = rec["latency"]
        print(f"run_id={rec['run_id']} latency={lat} peak_rss={rec['peak_rss_bytes']}B")
    elif args.cmd == "prune":
        from .prune import prune
        out = prune(args.model, args.method, args.target, seed=args.seed,
                    bench=not getattr(args, "no_bench", False))
        for r in out["rows"]:
            if r.get("stage") in ("post_prune", "post_finetune"):
                p = r.get("paths", {})
                print(f"run_id={r['run_id']} arm_id={r['arm_id']} stage={r['stage']} "
                      f"val_acc={r['val_acc']:.4f} test_acc={r['test_acc']:.4f} "
                      f"achieved_sparsity={r['achieved_sparsity']:.4f} "
                      f"status={r['status']}")
                for pname, pdata in (p or {}).items():
                    if not isinstance(pdata, dict):
                        continue
                    lat = {k: v for k, v in pdata.items()
                           if isinstance(v, dict) and "median_ms" in v}
                    note = pdata.get("note", "")
                    print(f"  path={pname} runtime_status={pdata.get('runtime_status')} "
                          f"latency={lat} peak_rss={pdata.get('peak_rss_bytes')} "
                          f"bytes={pdata.get('dense_bytes') or pdata.get('compressed_bytes')}")
                    if note:
                        print(f"    note: {note}")
        print(f"arm_id={out['arm_id']} status={out['status']} reason={out['reason']} "
              f"wall={out['wall_time_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
