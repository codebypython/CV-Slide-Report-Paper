"""Baseline and ablation architectures for scientific benchmarking.

Includes:
- Exp 0: Inception-v3 + Gender Concatenation (RSNA 2017 winner baseline)
- Exp 1: ConvNeXt Single-Branch + Gender Concatenation
- Exp 2: Swin Transformer Single-Branch + Gender Concatenation
- Exp 3: ConvNeXt + SwinT Naive Concat/Sum + Gender Concatenation
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class InceptionV3Baseline(nn.Module):
    """Replication of RSNA 2017 winning solution (Inception-v3 with gender concatenation)."""
    def __init__(
        self,
        model_name: str = "inception_v3",
        dropout_rate: float = 0.3,
        pretrained: bool = False
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = getattr(self.backbone, "num_features", 2048)

        # Gender projection
        self.gender_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )

        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(in_features + 32, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        img: torch.Tensor,
        gender: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        feat = self.backbone(img)  # [B, 2048]
        gender_feat = self.gender_proj(gender)  # [B, 32]
        fused = torch.cat([feat, gender_feat], dim=-1)  # [B, 2080]
        pred = self.regressor(fused).squeeze(-1)  # [B]
        return pred


class ConvNeXtSingleBranch(nn.Module):
    """Exp 1: Pure modern CNN inductive bias (ConvNeXt) evaluating micro-morphology alone."""
    def __init__(
        self,
        model_name: str = "convnext_small",
        dropout_rate: float = 0.3,
        pretrained: bool = False
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = getattr(self.backbone, "num_features", 768)

        self.gender_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(in_features + 32, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        img_conv: torch.Tensor,
        gender: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        feat = self.backbone(img_conv)
        gender_feat = self.gender_proj(gender)
        fused = torch.cat([feat, gender_feat], dim=-1)
        pred = self.regressor(fused).squeeze(-1)
        return pred


class SwinTSingleBranch(nn.Module):
    """Exp 2: Pure Hierarchical Vision Transformer (SwinT) evaluating macro-spatial attention alone."""
    def __init__(
        self,
        model_name: str = "swin_small_patch4_window7_224",
        dropout_rate: float = 0.3,
        pretrained: bool = False
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = getattr(self.backbone, "num_features", 768)

        self.gender_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(in_features + 32, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        img_swin: torch.Tensor,
        gender: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        feat = self.backbone(img_swin)
        gender_feat = self.gender_proj(gender)
        fused = torch.cat([feat, gender_feat], dim=-1)
        pred = self.regressor(fused).squeeze(-1)
        return pred


class NaiveDualBranchNet(nn.Module):
    """Exp 3: Dual-Branch ConvNeXt + SwinT combined via simple vector concatenation."""
    def __init__(
        self,
        convnext_name: str = "convnext_small",
        swin_name: str = "swin_small_patch4_window7_224",
        dropout_rate: float = 0.3,
        pretrained: bool = False
    ):
        super().__init__()
        self.convnext = timm.create_model(convnext_name, pretrained=pretrained, num_classes=0)
        convnext_dim = getattr(self.convnext, "num_features", 768)

        self.swin = timm.create_model(swin_name, pretrained=pretrained, num_classes=0)
        swin_dim = getattr(self.swin, "num_features", 768)

        self.gender_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )

        total_dim = convnext_dim + swin_dim + 32
        self.regressor = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        img_conv: torch.Tensor,
        img_swin: torch.Tensor,
        gender: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        f_c = self.convnext(img_conv)
        f_s = self.swin(img_swin)
        g = self.gender_proj(gender)

        fused = torch.cat([f_c, f_s, g], dim=-1)
        pred = self.regressor(fused).squeeze(-1)
        return pred
