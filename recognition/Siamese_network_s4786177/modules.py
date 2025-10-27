"""
modules.py
Student name: Jay Thakkar
Student number: s4786177
Description: Implements a ResNet50-based feature extractor with a small MLP head.
Stage: ResNet components only (no TripletLoss / classifier / wrapper yet).
"""

from __future__ import annotations  # use future annotations for forward refs in type hints
from typing import Optional

import torch                           # core tensor library
import torch.nn as nn                  # neural network modules
from torchvision import models         # pretrained ResNet50 (allowed to use from ed)


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
