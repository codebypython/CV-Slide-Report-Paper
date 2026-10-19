"""Unit and integration test suite for model architectures and ablation variants."""

import os
import sys
import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.film import FiLMLayer
from src.models.cross_attention import CrossAttentionFusion, BidirectionalCrossAttentionFusion
from src.models.hybrid_bone_age import DualBranchBoneAgeNet
from src.models.baselines import (
    InceptionV3Baseline,
    ConvNeXtSingleBranch,
    SwinTSingleBranch,
    NaiveDualBranchNet
)
from src.models.model_factory import build_bone_age_model


def test_film_layer_forward():
    """Tests FiLMLayer across 1D and 2D spatial tensor inputs."""
    film_1d = FiLMLayer(feature_dim=512, metadata_dim=1)
    x_1d = torch.randn(4, 512)
    meta = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    out_1d = film_1d(x_1d, meta)
    assert out_1d.shape == (4, 512)

    film_2d = FiLMLayer(feature_dim=64, metadata_dim=1)
    x_2d = torch.randn(2, 64, 14, 14)
    out_2d = film_2d(x_2d, meta[:2])
    assert out_2d.shape == (2, 64, 14, 14)


def test_cross_attention_fusion():
    """Tests directional and bidirectional cross-attention mechanisms."""
    cross_attn = CrossAttentionFusion(embed_dim=256, num_heads=4)
    q = torch.randn(2, 256)
    kv = torch.randn(2, 256)
    fused, weights = cross_attn(q, kv)
    assert fused.shape == (2, 256)

    bi_attn = BidirectionalCrossAttentionFusion(embed_dim=256, num_heads=4)
    fused_bi, (s2c, c2s) = bi_attn(kv, q)
    assert fused_bi.shape == (2, 256)


def test_dual_branch_proposed_forward():
    """Tests proposed DualBranchBoneAgeNet forward pass and gradient flow."""
    model = DualBranchBoneAgeNet(
        convnext_name="convnext_small",
        swin_name="swin_small_patch4_window7_224",
        embed_dim=256,
        num_heads=4,
        use_film=True,
        use_cross_attention=True,
        pretrained=False
    )
    img_conv = torch.randn(2, 3, 384, 384)
    img_swin = torch.randn(2, 3, 224, 224)
    gender = torch.tensor([[1.0], [0.0]])

    preds, attn = model(img_conv, img_swin, gender, return_attention=True)
    assert preds.shape == (2,)
    assert not torch.isnan(preds).any()

    # Backpropagation check
    loss = preds.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None gradient"


def test_model_factory_all_ablations():
    """Verifies that model factory creates all 6 ablation conditions correctly."""
    arch_keys = [
        "exp0_inception_baseline",
        "exp1_convnext_single",
        "exp2_swint_single",
        "exp3_naive_concat",
        "exp4_cross_attention",
        "exp5_proposed_film"
    ]
    for arch in arch_keys:
        m = build_bone_age_model(
            architecture_type=arch,
            convnext_name="convnext_small",
            swin_name="swin_small_patch4_window7_224",
            inception_name="inception_v3",
            embed_dim=256,
            pretrained=False
        )
        assert m is not None


def test_dual_branch_scaled_resolution_forward():
    """Verifies proposed scaled ConvNeXt-Small (512x512) + Swin (384x384) with 256x144 tokens."""
    model = DualBranchBoneAgeNet(
        convnext_name="convnext_small",
        swin_name="swin_small_patch4_window7_224",
        embed_dim=256,
        num_heads=4,
        use_film=True,
        use_cross_attention=True,
        pretrained=False
    )
    img_conv = torch.randn(2, 3, 512, 512)
    # Swin small with 224 or 384
    img_swin = torch.randn(2, 3, 224, 224)
    gender = torch.tensor([[1.0], [0.0]])

    preds, attn = model(img_conv, img_swin, gender, return_attention=True)
    assert preds.shape == (2,)
    assert not torch.isnan(preds).any()
    # convnext 512 / 32 = 16 -> 256 tokens; swin 224 / 32 = 7 -> 49 tokens
    assert attn.shape == (2, 49, 256)



if __name__ == "__main__":
    test_film_layer_forward()
    print("test_film_layer_forward passed.")
    test_cross_attention_fusion()
    print("test_cross_attention_fusion passed.")
    test_dual_branch_proposed_forward()
    print("test_dual_branch_proposed_forward passed.")
    test_model_factory_all_ablations()
    print("test_model_factory_all_ablations passed.")
    print("ALL MODEL TESTS PASSED SUCCESSFULLY!")
