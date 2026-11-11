"""Data loading: Fashion-MNIST and Cora (Planetoid public split)."""
from __future__ import annotations

import numpy as np
import torch
from torchvision import datasets as tv_datasets


def _fmnist_dataset(mode: str) -> tv_datasets.FashionMNIST:
    """mode in {'train','test'}: official 60k/10k sets, raw uint8 pixels."""
    from .util import DATA_DIR

    return tv_datasets.FashionMNIST(
        root=str(DATA_DIR), train=(mode == "train"), download=True
    )


def _images(ds: tv_datasets.FashionMNIST, idx: np.ndarray) -> torch.Tensor:
    """(N,1,28,28) float32 tensor for the given indices (raw 0..255)."""
    out = torch.empty((len(idx), 1, 28, 28), dtype=torch.float32)
    for row, i in enumerate(idx):
        out[row] = torch.from_numpy(np.asarray(ds[int(i)][0])).view(1, 28, 28)
    return out


def fmnist_split_tensors():
    """Return (train_x, train_y, val_x, val_y, test_x, test_y, mean, std) as float32
    tensors. Normalization mean/std computed on the 55k training split only (frozen).
    Indices come from the verified split files."""
    from .util import load_indices, set_threads

    set_threads()
    idx = load_indices("fmnist")
    train_ds = _fmnist_dataset("train")
    test_ds = _fmnist_dataset("test")
    tr_x = _images(train_ds, idx["train"])
    tr_y = torch.tensor([train_ds.targets[int(i)] for i in idx["train"]], dtype=torch.long)
    va_x = _images(train_ds, idx["val"])
    va_y = torch.tensor([train_ds.targets[int(i)] for i in idx["val"]], dtype=torch.long)
    te_x = _images(test_ds, idx["test"])
    te_y = torch.tensor([test_ds.targets[int(i)] for i in idx["test"]], dtype=torch.long)
    # Frozen: mean/std on the 55k training split only
    mean = tr_x.mean(dim=(0, 2, 3), keepdim=True)
    std = tr_x.std(dim=(0, 2, 3), keepdim=True)
    tr_x = (tr_x - mean) / std
    va_x = (va_x - mean) / std
    te_x = (te_x - mean) / std
    return tr_x, tr_y, va_x, va_y, te_x, te_y, mean, std


def load_cora():
    """Load Cora (Planetoid public split) and apply the frozen preprocessing:
    row-normalized features (x / (rowsum + 1e-12)). Returns a Data object with
    float32 features and the verified node index tensors lm_train/lm_val/lm_test."""
    from torch_geometric.datasets import Planetoid

    from .util import DATA_DIR, load_indices, set_threads

    set_threads()
    dataset = Planetoid(root=str(DATA_DIR / "Planetoid"), name="Cora")
    data = dataset[0]

    # Frozen: Cora features row-normalized (standard GCN preprocessing).
    x = data.x
    row_sum = x.sum(dim=1, keepdim=True)
    data.x = (x / (row_sum + 1e-12)).to(torch.float32)

    idx = load_indices("cora")
    data.lm_train = torch.tensor(idx["train"], dtype=torch.long)
    data.lm_val = torch.tensor(idx["val"], dtype=torch.long)
    data.lm_test = torch.tensor(idx["test"], dtype=torch.long)
    return data
