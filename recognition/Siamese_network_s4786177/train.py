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
from pathlib import Path

from dataset import get_isic2020_data_loaders, set_seed, DATA_ROOT
from modules import SiameseNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- metric-learning utilities (batch-hard triplet) ---
def _pairwise_distances(emb: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 pairwise distances between embeddings (assumes rows are embeddings).
    """
    dot = emb @ emb.t()                          # [B,B]
    sq = torch.diag(dot)                         # [B]
    dist2 = sq.unsqueeze(1) - 2 * dot + sq.unsqueeze(0)
    dist2 = torch.clamp(dist2, min=0.0)
    dist2.fill_diagonal_(0.0)
    return torch.sqrt(dist2 + 1e-12)

def _batch_hard_triplet_loss(
    emb: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Batch-hard triplet loss:
      hardest positive = farthest same-class
      hardest negative = closest different-class
    Returns mean over valid anchors. If batch has one class only, returns 0.
    """
    if emb.size(0) < 2 or labels.unique().numel() < 2:
        return emb.new_zeros(())
    d = _pairwise_distances(emb)                 # [B,B]
    B = emb.size(0)
    labels = labels.view(B, 1)
    same = (labels == labels.t())
    diff = ~same
    d_pos = d.clone()
    d_neg = d.clone()
    d_pos[~same] = -1e6
    d_pos.fill_diagonal_(-1e6)
    d_neg[same] = 1e6
    hardest_pos = d_pos.max(dim=1).values
    hardest_neg = d_neg.min(dim=1).values
    loss = torch.relu(hardest_pos - hardest_neg + margin)
    valid = (hardest_pos > -1e5) & (hardest_neg < 1e5)
    if valid.float().sum() == 0:
        return emb.new_zeros(())
    return loss[valid].mean()

def train_model(  
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    emb_dim: int = 128,
    drop: float = 0.6,
    save_path: str = "siamese_ce.pt",
    lambda_triplet: float = 1.0,
    triplet_margin: float = 0.2,
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
    # Save reports inside this Siamese network package directory so all
    # artifacts remain co-located with the model code.
    reports = (Path(__file__).parent / "reports").resolve()
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

            # Expecting model(x) -> (embeddings, logits). If your model returns only logits,
            # obtain embeddings via a helper like model.embed(x).
            emb, logits = model(x)
            loss_ce = ce(logits, y)
            loss_tri = _batch_hard_triplet_loss(emb, y, margin=triplet_margin)
            loss = loss_ce + lambda_triplet * loss_tri

            # backprop and optimizer step
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # Update running statistics for loss, accuracy and AUC
            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            # log components
            # train_ce_sum   += loss_ce.item() * bs
            # train_tri_sum  += loss_tri.item() * bs
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
                emb, logits = model(x)
                loss_ce = ce(logits, y)
                loss_tri = _batch_hard_triplet_loss(emb, y, margin=triplet_margin)
                loss = loss_ce + lambda_triplet * loss_tri

                # Update validation accumulators similar to training
                bs = x.size(0)
                val_loss_sum += loss.item() * bs
                # val_ce_sum   += loss_ce.item() * bs
                # val_tri_sum  += loss_tri.item() * bs
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

    # -------- final plots (loss + acc + auc) --------
    ep = np.arange(1, len(history["val_auc"]) + 1)

    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(ep, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(ep, history["val_loss"],   marker="o", label="Val Loss")
    plt.title("Loss over epochs"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.legend()
    plt.savefig(reports / "ce_loss.png", dpi=180, bbox_inches="tight"); plt.close()

    # Accuracy (already saved as acc.png during training: keep a final write too)
    plt.figure(figsize=(6,4))
    plt.plot(ep, history["train_acc"], marker="o", label="Train Acc")
    plt.plot(ep, history["val_acc"],   marker="o", label="Val Acc")
    plt.title("Accuracy over epochs"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(True); plt.legend()
    plt.savefig(reports / "ce_acc.png", dpi=180, bbox_inches="tight"); plt.close()

    # AUC
    plt.figure(figsize=(6,4))
    plt.plot(ep, history["train_auc"], marker="o", label="Train AUC")
    plt.plot(ep, history["val_auc"],   marker="o", label="Val AUC")
    plt.title("AUC over epochs"); plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.grid(True); plt.legend()
    plt.savefig(reports / "ce_auc.png", dpi=180, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    train_model()
