"""
modules.py
Student name: Jay Thakkar
Student number: s4786177
Description: Implements a ResNet50-based feature extractor with a small MLP head.
Stage: ResNet components + TripletLoss (no classifier/wrapper yet)
"""

from __future__ import annotations  # use future annotations for forward refs in type hints
from typing import Optional

import torch                            # core tensor library
import torch.nn as nn                   # neural network modules
from torchvision import models          # pretrained ResNet50 (allowed to use from ed)
import torch.nn.functional as F         # functional API for activations, losses, etc.

# -------------------------- helpers --------------------------
# optimal initialization method for neural networks that use ReLU activation functions.
# idea taken from https://towardsdatascience.com/kaiming-he-initialization-in-neural-networks-math-proof-73b9a0d845c4/
def _init_linear_stack(module: nn.Module, nonlinearity: str = "relu") -> None:
    """Kaiming init for all Linear layers in a stack."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# -------------------------- components --------------------------
class FeatureExtractor(nn.Module):
    """
    A ResNet50 backbone followed by a small MLP projection head that maps the
    penultimate ResNet features into a lower-dimensional embedding space.
    """
    def __init__(
        self,
        emb_dim: int = 128,
        drop: float = 0.5,
        pretrained: bool = True,
        l2_normalize: bool = False,
    ) -> None:
        super().__init__()
        self.l2_normalize = l2_normalize

        # Load ResNet50; replace the classification head with Identity.
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.resnet50(weights=weights)
        in_features = base.fc.in_features  # dimension of penultimate features
        base.fc = nn.Identity()            # strip ImageNet classifier
        self.base = base

        # Small MLP projection head: 2048 -> 512 -> 256 -> emb_dim
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, emb_dim),
        )

        # Kaiming initialization for linear layers in the MLP head
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image tensor -> embedding.
        """
        feats = self.base(x)     # [B, 2048] for ResNet50
        emb = self.head(feats)   # [B, emb_dim]
        if self.l2_normalize:
            # kept false for now
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

# https://github.com/shivsondhi/Triplet-Loss/blob/master/triplet_loss_functions.py for reference/ideation
class TripletLoss(nn.Module):
    """
    Triplet loss with margin:
        L = max(0, ||A-P||_2 - ||A-N||_2 + margin)
    Encourages the anchor to be closer to the positive than to the negative by at least 'margin'.
    """
    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, a: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        d_ap = torch.norm(a - p, p=2, dim=1)
        d_an = torch.norm(a - n, p=2, dim=1)
        return F.relu(d_ap - d_an + self.margin).mean()
    

class ClassifierHead(nn.Module):
    """
    Linear 2-class head operating on embeddings.
    Output: logits with shape [B, 2] (normal / melanoma).
    """
    def __init__(self, emb_dim: int = 128, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="linear")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)


class SiameseNet(nn.Module):
    """chore: yet to do
    """
    def __init__(self, emb_dim: int = 128, drop: float = 0.6, pretrained: bool = True) -> None:
        super().__init__()
        self.fe  = FeatureExtractor(emb_dim=emb_dim, drop=drop, pretrained=pretrained, l2_normalize=True)
        self.clf = ClassifierHead(emb_dim=emb_dim, num_classes=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.fe(x)
        logits = self.clf(emb)
        return emb, logits


