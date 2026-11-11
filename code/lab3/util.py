"""Shared utilities: frozen environment setup, splits, run logging, code hash."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Repo root (project root): <root>/code/lab3/util.py -> parents[2]
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SPLITS_DIR = ROOT / "splits"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "dense"
# P2: the results path is configurable via LAB3_RESULTS_DIR (tests point it at a
# tmp dir so pytest NEVER appends to the real results/runs.jsonl).
RESULTS_DIR = Path(os.environ.get("LAB3_RESULTS_DIR", str(ROOT / "results")))
CURVES_DIR = RESULTS_DIR / "curves"
RUNS_JSONL = RESULTS_DIR / "runs.jsonl"
BENCH_DIR = RESULTS_DIR / "bench"
FIGURES_DIR = ROOT / "figures"
CONFIGS_DIR = ROOT / "configs"

SPLIT_SEED = 12345
# B.1 (Part B): calibration subset — 1,024 TRAINING-split images, stratified
# 102-103 per class, drawn once with seed 12345. Labels are training labels.
CALIB_FILE = SPLITS_DIR / "fmnist_calib_idx.npy"
CALIB_N = 1024
SPLIT_FILES = {
    "fmnist": [
        SPLITS_DIR / "fmnist_train_idx.npy",
        SPLITS_DIR / "fmnist_val_idx.npy",
        SPLITS_DIR / "fmnist_test_idx.npy",
    ],
    "cora": [
        SPLITS_DIR / "cora_train_idx.npy",
        SPLITS_DIR / "cora_val_idx.npy",
        SPLITS_DIR / "cora_test_idx.npy",
    ],
}
SPLITS_SHA = SPLITS_DIR / "SPLITS.sha256"


def set_threads() -> None:
    """Frozen thread protocol: 4 intra-op threads, 1 inter-op thread."""
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # interop thread count can only be set before any parallel region starts
        pass


def set_seed(seed: int) -> None:
    """Frozen seed protocol: Python random, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_splits_hash() -> str:
    """SHA-256 over the six split index files (fixed order, name + array bytes)."""
    h = hashlib.sha256()
    for key in ("fmnist", "cora"):
        for path in SPLIT_FILES[key]:
            h.update(path.name.encode("utf-8"))
            h.update(_sha256_file(path).encode("ascii"))
    return h.hexdigest()


def make_calib_split() -> None:
    """B.1: create splits/fmnist_calib_idx.npy exactly once and APPEND its
    hash line to SPLITS.sha256 (never regenerate the other splits)."""
    if CALIB_FILE.exists():
        raise FileExistsError(
            f"{CALIB_FILE} already exists; the calibration subset is created once "
            "and never regenerated. Delete it manually only if you intend to reset.")
    np.random.seed(SPLIT_SEED)
    from .data import _fmnist_dataset
    train_ds = _fmnist_dataset("train")
    labels = np.asarray(train_ds.targets)
    train_idx = load_indices("fmnist")["train"]
    train_set = set(int(i) for i in train_idx)
    # 1024 = 10 * 102 + 4: the first 4 classes get 103, the rest 102.
    per = [103 if c < 4 else 102 for c in range(10)]
    assert sum(per) == CALIB_N
    calib = []
    for c in range(10):
        pool = np.asarray(sorted(i for i in train_idx if labels[int(i)] == c))
        assert set(labels[int(i)] for i in pool).issubset({c})
        calib.append(np.random.permutation(pool)[: per[c]])
    calib_idx = np.sort(np.concatenate(calib)).astype(np.int64)
    assert calib_idx.shape[0] == CALIB_N
    assert set(int(i) for i in calib_idx).issubset(train_set), "calib must be subset of train"
    np.save(CALIB_FILE, calib_idx)
    line = f"{_sha256_file(CALIB_FILE)}  {CALIB_FILE.name}\n"
    with open(SPLITS_SHA, "a") as f:
        f.write(line)
    print(f"calibration subset written to {CALIB_FILE}; appended hash to {SPLITS_SHA}")


def verify_splits() -> None:
    """Load all split files and verify SPLITS.sha256; never regenerates."""
    if not SPLITS_SHA.exists():
        raise FileNotFoundError(
            f"{SPLITS_SHA} not found. Run `uv run python -m lab3 make-splits` first. "
            "Splits are never regenerated automatically."
        )
    for key in SPLIT_FILES:
        for path in SPLIT_FILES[key]:
            if not path.exists():
                raise FileNotFoundError(f"missing split file {path}; run make-splits first")
    recorded = SPLITS_SHA.read_text().strip().split()[0]
    actual = compute_splits_hash()
    if actual != recorded:
        raise ValueError(
            f"split hash mismatch: recorded {recorded} != actual {actual}. "
            "Refusing to continue; run make-splits only if you intentionally reset splits."
        )


def load_indices(name: str) -> dict[str, np.ndarray]:
    verify_splits()
    out = {}
    files = {"train": SPLIT_FILES[name][0], "val": SPLIT_FILES[name][1], "test": SPLIT_FILES[name][2]}
    for k, p in files.items():
        out[k] = np.load(p)
    return out


def code_hash() -> str:
    """SHA-256 over all files under code/ (git-free: pure file-content hash)."""
    h = hashlib.sha256()
    code_dir = ROOT / "code"
    for path in sorted(code_dir.rglob("*")):
        if path.is_file():
            h.update(str(path.relative_to(code_dir)).encode("utf-8"))
            h.update(path.read_bytes())
    return h.hexdigest()


def _ensure_dirs() -> None:
    # P2: tests redirect LAB3_RESULTS_DIR at a tmp dir; only the results tree is
    # created there. DATA_DIR / SPLITS_DIR / FIGURES_DIR already exist in the repo
    # (never created under a redirected results root).
    for d in (RESULTS_DIR, CURVES_DIR, BENCH_DIR):
        d.mkdir(parents=True, exist_ok=True)


def new_run_id(architecture: str, method: str = "dense") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{architecture}-{method}-{ts}-{uuid.uuid4().hex[:6]}"


def write_run_record(rec: dict) -> None:
    """Append one JSON line to results/runs.jsonl. Never rewrites or deletes."""
    _ensure_dirs()
    with open(RUNS_JSONL, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def prior_bench_run_ids(architecture: str, seed: int = 0) -> list[str]:
    """run_ids of all earlier successful `bench` records for this (arch, seed), in
    file order. Used to fill `supersedes` on a corrected re-bench. The log is
    append-only: earlier lines are never edited or deleted."""
    out = []
    if not RUNS_JSONL.exists():
        return out
    with open(RUNS_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("command") == "bench" and r.get("status") == "ok"
                    and r.get("architecture") == architecture and r.get("seed") == seed):
                out.append(r["run_id"])
    return out


def versions() -> dict:
    import platform

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def base_record(architecture: str, command: str) -> dict:
    return {
        "run_id": new_run_id(architecture),
        # B2: pilot phase of the matrix. "pilot" for the Part B pilot matrix
        # (and its dense_ft controls); the confirmatory reruns will carry
        # "confirm". Part A records have phase=None (pre-B2 schema). Every NEW
        # record after B2 carries the field; see results/SCHEMA.md.
        "phase": None,
        "status": "ok",
        "reason": None,
        "part": "A",
        "architecture": architecture,
        "method": "dense",
        "target_sparsity": 0.0,
        "achieved_sparsity": 0.0,
        "stored_precision": "fp32",
        "compute_precision": "fp32",
        "accumulator_precision": "fp32",
        "seed": None,
        "train_epochs": None,
        "finetune_epochs": 0,
        "best_epoch": None,
        "val_acc": None,
        "val_loss": None,
        "test_acc": None,
        "total_params": None,
        "nonzero_params": None,
        "dense_bytes": None,
        "raw_tensor_bytes": None,
        # P1: compressed_bytes is reserved for the compact-export artifact
        # (Part B path 2). gzip_bytes is the gzip-9 size of the dense state_dict
        # file and is reported for every run (dense included).
        "compressed_bytes": None,
        "gzip_bytes": None,
        "est_macs": None,
        "est_macs_nonzero": None,
        "est_macs_sparse_input": None,
        "gcn_norm_macs": None,
        "peak_rss_bytes": None,
        "rss_before_model_bytes": None,
        "model_rss_delta_bytes": None,
        "latency": {},
        "parent_run_id": None,
        "supersedes": [],
        "wall_time_s": None,
        "code_hash": code_hash(),
        "versions": versions(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
    }


def finalize_record(rec: dict, *, status: str | None = None, reason: str | None = None) -> dict:
    if status is not None:
        rec["status"] = status
        rec["reason"] = reason
    return rec


def run_wall_timer(rec: dict, t0: float) -> dict:
    rec["wall_time_s"] = round(time.perf_counter() - t0, 3)
    return rec
