"""
train.py
Student name: Jay Thakkar
Student number: s4786177
Description: 1 epoch train script with minimal features to verify training works.
Stage: Basically barebones
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from dataset import get_isic2020_data_loaders, set_seed
from modules import SiameseNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-4,
    emb_dim: int = 128,
    drop: float = 0.6,
) -> None:
    """Train for exactly one epoch and print train/val accuracy."""
    set_seed(42)
    train_loader, val_loader, _ = get_isic2020_data_loaders(bs=batch_size, workers=2, seed=42)

    model = SiameseNet(emb_dim=emb_dim, drop=drop, pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce  = nn.CrossEntropyLoss()
    
    history = {"train_acc": [], "val_acc": []} # track values

    # ---------------- train (based on epoch param) ----------------
    for epoch in range(1, epochs + 1):
        model.train()
        train_correct, train_count = 0, 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            _, logits = model(x)
            loss = ce(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_correct += (logits.argmax(1) == y).sum().item()
            train_count   += x.size(0)

        # ------ validate ------
        model.eval()
        val_correct, val_count = 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                _, logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_count   += x.size(0)

        tr_acc = train_correct / max(1, train_count)
        va_acc = val_correct   / max(1, val_count)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch}:: train accuracy: {tr_acc:.3f} | val accuracy: {va_acc:.3f}")


if __name__ == "__main__":
    train_one_epoch()
