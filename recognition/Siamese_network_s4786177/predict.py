"""
predict.py
Student name: Jay Thakkar
Student number: s4786177
Description: Evaluation, inference, and visualization utilities for image pairs.
Stage: Loading best checkpoint + inference on test set + print accuracy and auc roc (no vis, plot, )
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from dataset import get_isic2020_data_loaders, set_seed
from modules import SiameseNet


def evaluate_on_test(
    checkpoint_path: str = "siamese_ce.pt",
    batch_size: int = 32,
    emb_dim: int = 128,
    drop: float = 0.6,
    seed: int = 42,
) -> Tuple[float, float]:
    """chore: to do later
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data: only need the test loader here
    _, _, test_loader = get_isic2020_data_loaders(bs=batch_size, workers=0 if device.type=="cpu" else 2, seed=seed)

    # load the model: init architecture, load checkpoint weights
    model = SiameseNet(emb_dim=emb_dim, drop=drop, pretrained=False).to(device)
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path.resolve()}\n"
            "Train first (train.py) to produce the best model checkpoint."
        )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Inference loop
    correct, count = 0, 0
    all_probs: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            _, logits = model(x) # [B,2] shaep
            preds = logits.argmax(1) # [B] shape
            correct += (preds == y).sum().item()
            count   += y.size(0)

            # class-1 (melanoma) probability for AUC ROC
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(y.detach().cpu().numpy().tolist())

    # Metrics
    test_acc = correct / max(1, count)
    try:
        test_auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        test_auc = float("nan")  # handle edge-cases gracefully

    print(f"[Test] Accuracy: {test_acc:.3f} | AUC ROC: {test_auc:.3f}")
    # ---- plots: ROC + Confusion Matrix ----
    reports = (DATA_ROOT.parent / "reports").resolve()
    reports.mkdir(parents=True, exist_ok=True)

    # ROC curve
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(all_labels, all_probs)
    plt.title("Test ROC Curve")
    plt.grid(True, alpha=0.3)
    plt.savefig(reports / "test_roc.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Confusion matrix (at argmax-threshold)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (argmax)")
    plt.colorbar()
    plt.xticks([0, 1], ["Normal (0)", "Melanoma (1)"])
    plt.yticks([0, 1], ["Normal (0)", "Melanoma (1)"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(reports / "test_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close()

    return test_acc, test_auc


if __name__ == "__main__":
    evaluate_on_test()
