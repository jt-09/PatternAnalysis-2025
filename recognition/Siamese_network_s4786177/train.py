"""
train.py
Student name: Jay Thakkar
Student number: s4786177
Description: CE trainer with full metrics (loss + AUC), plots, and save-best checkpoint.
Stage: Production CE trainer (Triplet joint loss comes later).
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from dataset import get_isic2020_data_loaders, set_seed, DATA_ROOT
from modules import SiameseNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(  
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    emb_dim: int = 128,
    drop: float = 0.6,
    save_path: str = "siamese_ce.pt",
) -> None:
    """chore: will do later
    """
    set_seed(42)
    train_loader, val_loader, _ = get_isic2020_data_loaders(bs=batch_size, workers=2, seed=42)
    print(f"[info] device={device} | pretrained=True")
    model = SiameseNet(emb_dim=emb_dim, drop=drop, pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce  = nn.CrossEntropyLoss()

    # Model and optimizer setup: create the SiameseNet (with optional
    # pretrained backbone), move it to the selected device, and prepare
    # an Adam optimizer along with a cross-entropy loss for the
    # classification task.
    
    history = {
        "train_loss": [], "train_acc": [], "train_auc": [],
        "val_loss":   [], "val_acc":   [], "val_auc":   [],
    }

    # reporting and checkpointing: create a small history dict to store
    # loss/acc/auc per epoch for both train and validation. Also ensure
    # a `reports` directory exists next to the DATA_ROOT where plots will
    # be saved. Track the best validation AUC to decide which checkpoint
    # to persist.
    reports = (DATA_ROOT.parent / "reports").resolve()
    reports.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0  # best Validation AUC so far

    # ---------------- train (multi-epoch) ----------------
    for epoch in range(1, epochs + 1):
        model.train()
        train_correct, train_count = 0, 0
        train_loss_sum = 0.0
        probs_tr, y_tr = [], []

        # Epoch-level training: set the model to train mode and reset
        # accumulators for loss, correct predictions and probability
        # lists used to compute AUC at the end of the epoch.

        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward pass: the model returns embeddings and logits; use
            # the logits for classification loss and metric computation.
            _, logits = model(x)
            loss = ce(logits, y)

            # backprop and optimizer step
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # Update running statistics for loss, accuracy and AUC
            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            train_correct  += (logits.argmax(1) == y).sum().item()
            train_count    += bs

            # Collect probabilities and labels for epoch-level AUC
            probs_tr += list(torch.softmax(logits, 1)[:, 1].detach().cpu().numpy())
            y_tr     += list(y.detach().cpu().numpy())

        tr_loss = train_loss_sum / max(1, train_count)
        tr_acc  = train_correct  / max(1, train_count)
        try:
            tr_auc = roc_auc_score(y_tr, probs_tr)
        except Exception:
            # AUC computation can fail if only one class is present in
            # the epoch; handle that gracefully by using NaN.
            tr_auc = float("nan")

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_auc"].append(tr_auc)


        # ------ validate ------
        # Switch to evaluation mode and accumulate validation metrics
        model.eval()
        val_correct, val_count = 0, 0
        val_loss_sum = 0.0
        probs_va, y_va = [], []

        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                _, logits = model(x)
                loss = ce(logits, y)

                # Update validation accumulators similar to training
                bs = x.size(0)
                val_loss_sum += loss.item() * bs
                val_correct  += (logits.argmax(1) == y).sum().item()
                val_count    += bs

                # Collect predicted probabilities and labels for AUC
                probs_va += list(torch.softmax(logits, 1)[:, 1].cpu().numpy())
                y_va     += list(y.cpu().numpy())

        va_loss = val_loss_sum / max(1, val_count)
        va_acc  = val_correct  / max(1, val_count)
        try:
            va_auc = roc_auc_score(y_va, probs_va)
        except Exception:
            # Same AUC caveat on validation set: fallback to NaN if
            # computation isn't possible (e.g., single-class validation
            # batch or other corner cases).
            va_auc = float("nan")

        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_auc"].append(va_auc)

        print(
            f"Epoch {epoch}: "
            f"Train Loss={tr_loss:.3f} Acc={tr_acc:.3f} AUC={tr_auc:.3f} | "
            f"Val Loss={va_loss:.3f} Acc={va_acc:.3f} AUC={va_auc:.3f}"
        )
        
        # save best checkpoint by Validation AUC
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save(model.state_dict(), save_path)
            print(f"saved best model (using best auc) into -> {save_path}")

        # If the validation AUC improved this epoch, persist the model
        # weights so the best-performing checkpoint is available after
        # training completes.
        
        # ---- accuracy over completed epochs ----
        # Save an intermediate accuracy plot each epoch so progress can be
        # observed while training is running. This produces a simple
        # visualization of train vs validation accuracy over epochs.
        ep = np.arange(1, len(history["train_acc"]) + 1)
        plt.figure(figsize=(6,4))
        plt.plot(ep, history["train_acc"], marker="o", label="Train Acc")
        plt.plot(ep, history["val_acc"],   marker="o", label="Val Acc")
        plt.title("Accuracy over epochs")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(True); plt.legend()
        plt.savefig(reports / "acc.png", dpi=180, bbox_inches="tight")
        plt.close()

    


if __name__ == "__main__":
    train_model()
