"""
predict.py
Student name: Jay Thakkar
Student number: s4786177
Description: Evaluation, inference, and visualization utilities for image pairs.
Stage: Loading best checkpoint + inference on test set + print accuracy and auc roc (no vis, plot, )
"""


from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    roc_auc_score,
    RocCurveDisplay,
    confusion_matrix,
    roc_curve,
)
from sklearn.manifold import TSNE

from dataset import get_isic2020_data_loaders, set_seed, DATA_ROOT
from modules import SiameseNet

def _youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute threshold that maximizes Youden's J = TPR - FPR."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[np.argmax(j)])
def evaluate_on_test(
    checkpoint_path: str = "siamese_ce.pt",
    batch_size: int = 32,
    emb_dim: int = 128,
    drop: float = 0.6,
    seed: int = 42,
    plot_tsne: bool = True,
) -> Tuple[float, float]:
    """chore: to do later
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data: only need the test loader here
    _, _, test_loader = get_isic2020_data_loaders(
        bs=batch_size, workers=(0 if device.type == "cpu" else 2), seed=seed
    )

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
    all_preds_argmax: list[int] = []
    all_embs: list[np.ndarray] = []

    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            emb, logits = model(x)              # [B,emb], [B,2]
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(1)            # [B]

            correct += (preds == y).sum().item()
            count   += y.size(0)

            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(y.detach().cpu().numpy().tolist())
            all_preds_argmax.extend(preds.detach().cpu().numpy().tolist())
            all_embs.append(emb.detach().cpu().numpy())

    # Metrics @ argmax
    test_acc = correct / max(1, count)
    try:
        test_auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        test_auc = float("nan")  # handle edge-cases gracefully

    y_true = np.asarray(all_labels, dtype=int)
    y_prob = np.asarray(all_probs, dtype=float)
    thr_youden = _youden_threshold(y_true, y_prob)
    y_pred_thr = (y_prob >= thr_youden).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thr, labels=[0, 1]).ravel()
    spec = tn / max(1, (tn + fp))  # specificity (TNR)
    sens = tp / max(1, (tp + fn))  # sensitivity (TPR)

    print(f"[Test] AUC: {test_auc:.3f}")
    print(f"[Test] Acc@argmax: {test_acc:.3f}")
    print(f"[Test] Thr[Youden]: {thr_youden:.4f} | Spec: {spec:.3f} | Sens: {sens:.3f}")

    # ---- plots: ROC + Confusion Matrix ----
    # Save reports inside this Siamese network package directory so all
    # artifacts remain co-located with the model code.
    reports = (Path(__file__).parent / "reports").resolve()
    reports.mkdir(parents=True, exist_ok=True)

    # ROC curve
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("Test ROC Curve")
    plt.grid(True, alpha=0.3)
    plt.savefig(reports / "test_roc.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Confusion matrix at argmax (prob→class by argmax logits)
    cm_argmax = confusion_matrix(y_true, np.asarray(all_preds_argmax), labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm_argmax, cmap="Blues")
    plt.title("Confusion Matrix (argmax)")
    plt.colorbar()
    plt.xticks([0, 1], ["Normal (0)", "Melanoma (1)"])
    plt.yticks([0, 1], ["Normal (0)", "Melanoma (1)"])
    for (i, j), v in np.ndenumerate(cm_argmax):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(reports / "test_confusion_matrix_argmax.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Confusion matrix at Youden threshold
    cm_thr = confusion_matrix(y_true, y_pred_thr, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm_thr, cmap="Purples")
    plt.title("Confusion Matrix (Youden threshold)")
    plt.colorbar()
    plt.xticks([0, 1], ["Normal (0)", "Melanoma (1)"])
    plt.yticks([0, 1], ["Normal (0)", "Melanoma (1)"])
    for (i, j), v in np.ndenumerate(cm_thr):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(reports / "test_confusion_matrix_threshold.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Optional: t-SNE of test embeddings for qualitative separation
    if plot_tsne:
        try:
            X = np.concatenate(all_embs, axis=0)
            y = y_true
            tsne = TSNE(n_components=2, init="random", perplexity=30, learning_rate="auto")
            X2 = tsne.fit_transform(X)

            plt.figure(figsize=(6, 5))
            plt.scatter(X2[y == 0, 0], X2[y == 0, 1], s=6, alpha=0.7, label="Normal (0)")
            plt.scatter(X2[y == 1, 0], X2[y == 1, 1], s=9, alpha=0.9, label="Melanoma (1)")
            plt.title("t-SNE of Test Embeddings")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(reports / "test_tsne.png", dpi=180, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"[warn] t-SNE failed: {e}")

    return test_acc, test_auc


if __name__ == "__main__":
    evaluate_on_test()