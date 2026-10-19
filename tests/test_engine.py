"""Unit tests for training engine, unfreezing strategy, and bootstrap statistics."""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.hybrid_bone_age import DualBranchBoneAgeNet
from src.engine.trainer import train_one_epoch, evaluate_epoch, set_backbone_freeze, get_optimizer_param_groups
from src.engine.evaluator import compute_bootstrap_ci, is_pubertal_acceleration_cohort
from sklearn.metrics import mean_absolute_error


class MockDataset:
    def __init__(self, size=8):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_c = torch.randn(3, 384, 384)
        img_s = torch.randn(3, 224, 224)
        gender = torch.tensor([1.0 if idx % 2 == 0 else 0.0])
        target = torch.tensor(120.0 + idx * 2.0)
        meta = {"id": idx, "male": idx % 2 == 0, "bone_age": 120.0, "age_bracket": "Childhood"}
        return img_c, img_s, gender, target, meta


def test_training_and_eval_step():
    """Tests one training epoch and one eval epoch with mock data."""
    device = torch.device("cpu")
    model = DualBranchBoneAgeNet(
        convnext_name="convnext_small",
        swin_name="swin_small_patch4_window7_224",
        embed_dim=128,
        num_heads=2,
        pretrained=False
    ).to(device)

    dataset = MockDataset(size=4)
    loader = DataLoader(dataset, batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.SmoothL1Loss(beta=1.0)

    train_loss, train_mae = train_one_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        scaler=None,
        device=device,
        criterion=criterion,
        gradient_accumulation_steps=1
    )
    assert not np.isnan(train_loss)
    assert not np.isnan(train_mae)

    val_loss, val_mae = evaluate_epoch(
        model=model,
        dataloader=loader,
        device=device,
        criterion=criterion
    )
    assert not np.isnan(val_loss)
    assert not np.isnan(val_mae)


def test_gradual_unfreeze_and_param_groups():
    """Verifies that backbone freeze correctly toggles requires_grad."""
    model = DualBranchBoneAgeNet(
        convnext_name="convnext_small",
        swin_name="swin_small_patch4_window7_224",
        embed_dim=128,
        pretrained=False
    )

    # Freeze backbones
    set_backbone_freeze(model, freeze=True)
    for p in model.convnext.parameters():
        assert not p.requires_grad
    for p in model.swin.parameters():
        assert not p.requires_grad
    for p in model.regressor.parameters():
        assert p.requires_grad

    # Unfreeze backbones
    set_backbone_freeze(model, freeze=False)
    for p in model.convnext.parameters():
        assert p.requires_grad
    for p in model.swin.parameters():
        assert p.requires_grad

    # Verify param groups configuration
    groups = get_optimizer_param_groups(model, lr_backbone=5e-5, lr_head=2e-4)
    assert len(groups) == 2
    assert groups[0]["lr"] == 5e-5
    assert groups[1]["lr"] == 2e-4


def test_bootstrap_ci_computation():
    """Verifies 1000x bootstrap CI calculation."""
    y_true = np.array([120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0])
    y_pred = np.array([122.0, 128.0, 143.0, 149.0, 161.0, 168.0, 182.0, 189.0])

    point, low, high = compute_bootstrap_ci(y_true, y_pred, mean_absolute_error, n_bootstrap=200, seed=42)
    assert low <= point <= high
    assert point == mean_absolute_error(y_true, y_pred)


def test_pubertal_cohort_filter():
    """Verifies clinical pubertal growth spurt age criteria."""
    # Female: 11-14y (132 to 168 months)
    assert is_pubertal_acceleration_cohort(140.0, is_male=False)
    assert not is_pubertal_acceleration_cohort(120.0, is_male=False)

    # Male: 12-15y (144 to 180 months)
    assert is_pubertal_acceleration_cohort(150.0, is_male=True)
    assert not is_pubertal_acceleration_cohort(135.0, is_male=True)


if __name__ == "__main__":
    test_training_and_eval_step()
    print("test_training_and_eval_step passed.")
    test_gradual_unfreeze_and_param_groups()
    print("test_gradual_unfreeze_and_param_groups passed.")
    test_bootstrap_ci_computation()
    print("test_bootstrap_ci_computation passed.")
    test_pubertal_cohort_filter()
    print("test_pubertal_cohort_filter passed.")
    print("ALL ENGINE TESTS PASSED SUCCESSFULLY!")
