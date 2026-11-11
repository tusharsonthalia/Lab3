"""Test (d): split files load, hashes match, splits are disjoint, fmnist val
is 500 per class. Creates the session's real split files if missing (same
frozen procedure as make-splits, split seed 12345)."""
import numpy as np

import lab3.util as U
from lab3.util import load_indices, verify_splits, SPLIT_SEED


def _make_splits_once():
    """Create the session's real split files if missing (tests need them)."""
    existing = [p for p in U.SPLIT_FILES["fmnist"] + U.SPLIT_FILES["cora"] + [U.SPLITS_SHA]]
    if all(p.exists() for p in existing):
        return
    U.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SPLIT_SEED)

    from lab3.data import _fmnist_dataset
    train_ds = _fmnist_dataset("train")
    labels = np.asarray(train_ds.targets)
    val_idx = []
    for cls in range(10):
        pool = np.flatnonzero(labels == cls)
        val_idx.append(np.random.permutation(pool)[:500])
    val_idx = np.sort(np.concatenate(val_idx))
    train_idx = np.setdiff1d(np.arange(60000), val_idx, assume_unique=True)
    test_idx = np.arange(10000)

    np.save(U.SPLITS_DIR / "fmnist_train_idx.npy", train_idx.astype(np.int64))
    np.save(U.SPLITS_DIR / "fmnist_val_idx.npy", val_idx.astype(np.int64))
    np.save(U.SPLITS_DIR / "fmnist_test_idx.npy", test_idx.astype(np.int64))

    from lab3.cli import _cora_masks
    tr_m, va_m, te_m = _cora_masks()
    np.save(U.SPLITS_DIR / "cora_train_idx.npy", np.flatnonzero(tr_m.numpy()).astype(np.int64))
    np.save(U.SPLITS_DIR / "cora_val_idx.npy", np.flatnonzero(va_m.numpy()).astype(np.int64))
    np.save(U.SPLITS_DIR / "cora_test_idx.npy", np.flatnonzero(te_m.numpy()).astype(np.int64))

    U.SPLITS_SHA.write_text(U.compute_splits_hash() + "\n")


def test_split_files_load_and_hash():
    _make_splits_once()
    verify_splits()
    fm = load_indices("fmnist")
    co = load_indices("cora")
    assert fm["train"].shape[0] == 55000
    assert fm["val"].shape[0] == 5000
    assert fm["test"].shape[0] == 10000
    assert co["train"].shape[0] == 140
    assert co["val"].shape[0] == 500
    assert co["test"].shape[0] == 1000


def test_splits_disjoint():
    """Within-dataset disjointness. Note: fmnist train/val are indices into the
    60k train set while fmnist test indices refer to the separate 10k test set,
    so cross-set index overlap is by design and not checked."""
    _make_splits_once()
    fm = load_indices("fmnist")
    co = load_indices("cora")
    # fmnist: train and val both index the 60k train pool -> must be disjoint
    # and together cover all 60k train images exactly once.
    t, v = fm["train"], fm["val"]
    assert len(np.intersect1d(t, v)) == 0, "fmnist train/val overlap"
    union = np.union1d(t, v)
    assert union.shape[0] == 60000 and union.min() == 0 and union.max() == 59999
    assert np.array_equal(np.sort(np.concatenate([t, v])), np.arange(60000))
    # fmnist test: official 10k set
    assert np.array_equal(fm["test"], np.arange(10000))
    # cora: all three index the same 2708-node graph -> pairwise disjoint
    ct, cv, cte = co["train"], co["val"], co["test"]
    assert len(np.intersect1d(ct, cv)) == 0, "cora train/val overlap"
    assert len(np.intersect1d(ct, cte)) == 0, "cora train/test overlap"
    assert len(np.intersect1d(cv, cte)) == 0, "cora val/test overlap"
    assert np.union1d(ct, np.union1d(cv, cte)).shape[0] == 1640


def test_fmnist_val_500_per_class():
    _make_splits_once()
    from lab3.data import _fmnist_dataset
    fm = load_indices("fmnist")
    labels = np.asarray(_fmnist_dataset("train").targets)
    for cls in range(10):
        assert int((labels[fm["val"]] == cls).sum()) == 500, f"val class {cls} != 500"
