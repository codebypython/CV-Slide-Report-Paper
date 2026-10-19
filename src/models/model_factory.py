"""Factory module to instantiate models across all stages and ablation experiments."""

import torch.nn as nn
from typing import Dict, Any

from .hybrid_bone_age import DualBranchBoneAgeNet
from .baselines import (
    InceptionV3Baseline,
    ConvNeXtSingleBranch,
    SwinTSingleBranch,
    NaiveDualBranchNet
)


def build_bone_age_model(
    architecture_type: str,
    convnext_name: str = "convnext_small",
    swin_name: str = "swin_small_patch4_window7_224",
    inception_name: str = "inception_v3",
    embed_dim: int = 512,
    num_heads: int = 8,
    dropout_rate: float = 0.3,
    pretrained: bool = False
) -> nn.Module:
    """Builds model instance matching the specified experiment architecture."""
    arch = architecture_type.lower()

    if arch in ["exp0", "exp0_inception_baseline", "inception_baseline"]:
        return InceptionV3Baseline(
            model_name=inception_name,
            dropout_rate=dropout_rate,
            pretrained=pretrained
        )
    elif arch in ["exp1", "exp1_convnext_single", "convnext_single"]:
        return ConvNeXtSingleBranch(
            model_name=convnext_name,
            dropout_rate=dropout_rate,
            pretrained=pretrained
        )
    elif arch in ["exp2", "exp2_swint_single", "swint_single", "swin_single"]:
        return SwinTSingleBranch(
            model_name=swin_name,
            dropout_rate=dropout_rate,
            pretrained=pretrained
        )
    elif arch in ["exp3", "exp3_naive_concat", "naive_concat"]:
        return NaiveDualBranchNet(
            convnext_name=convnext_name,
            swin_name=swin_name,
            dropout_rate=dropout_rate,
            pretrained=pretrained
        )
    elif arch in ["exp4", "exp4_cross_attention", "cross_attention_only"]:
        return DualBranchBoneAgeNet(
            convnext_name=convnext_name,
            swin_name=swin_name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            use_film=False,
            use_cross_attention=True,
            pretrained=pretrained
        )
    elif arch in ["exp5", "exp5_proposed_film", "proposed_film_cross_attn", "proposed"]:
        return DualBranchBoneAgeNet(
            convnext_name=convnext_name,
            swin_name=swin_name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            use_film=True,
            use_cross_attention=True,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown architecture_type: {architecture_type}")
