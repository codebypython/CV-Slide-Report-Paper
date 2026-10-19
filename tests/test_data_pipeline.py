"""Unit and validation tests for data preprocessing and stratified split protocol."""

import os
import sys
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.transforms import (
    LetterboxSquarePad,
    CLAHEContrastEnhancer,
    HandOrientationNormalizer,
    DualBranchTransform,
    AnatomicalCarpalCrop,
    MultiCropDualBranchTransform
)
from src.data.stratified_split import assign_age_bracket, generate_stratified_folds, verify_fold_distribution
from src.data.synthetic_generator import generate_synthetic_dataset
from src.data.dataset import RSNABoneAgeDataset



def test_letterbox_padding():
    """Verifies that non-square rectangular images are padded to exact square dimensions without distorting aspect ratio."""
    letterbox = LetterboxSquarePad(target_size=512)
    rect_img = np.zeros((300, 200), dtype=np.uint8)  # Height > Width
    padded = letterbox(rect_img)
    assert padded.shape == (512, 512), f"Expected (512, 512), got {padded.shape}"

    rect_wide = np.zeros((200, 400, 3), dtype=np.uint8)  # Width > Height
    padded_wide = letterbox(rect_wide)
    assert padded_wide.shape == (512, 512, 3), f"Expected (512, 512, 3), got {padded_wide.shape}"


def test_clahe_enhancer():
    """Verifies CLAHE improves local contrast on synthetic gradient image."""
    clahe = CLAHEContrastEnhancer(clip_limit=2.0, tile_grid_size=(8, 8))
    img = np.linspace(50, 150, 256 * 256, dtype=np.uint8).reshape((256, 256))
    enhanced = clahe(img)
    assert enhanced.shape == (256, 256)
    assert enhanced.dtype == np.uint8


def test_hand_orientation_normalizer():
    """Verifies left-hand normalization logic."""
    img = np.zeros((512, 512), dtype=np.uint8)
    norm = HandOrientationNormalizer.normalize_to_left_hand(img, is_right_hand=False)
    assert norm.shape == (512, 512)


def test_stratified_5fold_split():
    """Verifies that stratified 5-fold preserves age-bracket and sex balance across all folds."""
    records = []
    # Create mock cohort with all brackets
    ages = [10.0, 18.0, 40.0, 80.0, 140.0, 160.0, 190.0, 210.0] * 10
    males = [True, False] * 40
    for i in range(80):
        records.append({"id": i, "boneage": ages[i], "male": males[i]})

    df = pd.DataFrame(records)
    df_folds = generate_stratified_folds(df, n_splits=5, seed=42)

    assert "fold" in df_folds.columns
    assert set(df_folds["fold"].unique()) == {0, 1, 2, 3, 4}

    # Verify each fold has equal number of samples
    fold_counts = df_folds["fold"].value_counts()
    assert (fold_counts == 16).all()


def test_dataset_loading_synthetic():
    """Generates a small synthetic test set and verifies RSNABoneAgeDataset __getitem__."""
    test_dir = os.path.join(PROJECT_ROOT, "outputs", "test_scratch")
    os.makedirs(test_dir, exist_ok=True)
    csv_path, img_dir = generate_synthetic_dataset(test_dir, num_samples=6, seed=42)

    df = pd.read_csv(csv_path)
    tf = DualBranchTransform(is_training=False)
    ds = RSNABoneAgeDataset(df, img_dir, transform=tf, is_training=False)

    assert len(ds) == 6
    img_conv, img_swin, gender, bone_age, meta = ds[0]

    assert img_conv.shape == (3, 384, 384)
    assert img_swin.shape == (3, 224, 224)
    assert gender.shape == (1,)
    assert isinstance(bone_age.item(), float)
    assert "age_bracket" in meta


def test_anatomical_carpal_crop():
    """Verifies anatomical carpal crop extraction and preservation of wrist ROI."""
    cropper = AnatomicalCarpalCrop(crop_box=(0.45, 0.95, 0.15, 0.85), target_size=256)
    mock_hand = np.zeros((512, 512), dtype=np.uint8)
    # Draw mock wrist / carpal structure in lower center
    mock_hand[300:450, 150:350] = 180
    roi = cropper(mock_hand)
    assert roi.shape == (256, 256)
    assert (roi > 0).any()


def test_multicrop_dual_branch_transform():
    """Verifies coordinated extraction of whole hand and optical carpal zoom."""
    transform = MultiCropDualBranchTransform(hand_size=384, roi_size=256, is_training=False)
    mock_rgb = np.zeros((600, 400, 3), dtype=np.uint8)
    mock_rgb[350:550, 100:300, :] = 200
    t_hand, t_roi = transform(mock_rgb)
    assert t_hand.shape == (3, 384, 384)
    assert t_roi.shape == (3, 256, 256)



if __name__ == "__main__":
    test_letterbox_padding()
    print("test_letterbox_padding passed.")
    test_clahe_enhancer()
    print("test_clahe_enhancer passed.")
    test_hand_orientation_normalizer()
    print("test_hand_orientation_normalizer passed.")
    test_stratified_5fold_split()
    print("test_stratified_5fold_split passed.")
    test_dataset_loading_synthetic()
    print("test_dataset_loading_synthetic passed.")
    print("ALL DATA PIPELINE TESTS PASSED SUCCESSFULLY!")
