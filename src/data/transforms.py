"""Anatomically preserving data preprocessing and transforms for pediatric bone age radiographs.

Key scientific tenets implemented:
1. Aspect-Preserving Square Letterboxing: Prevents artificial morphometric distortion of long bones.
2. Canonical Hand Orientation Normalization: Standardizes right-hand radiographs to the left-hand atlas convention.
3. CLAHE (Contrast Limited Adaptive Histogram Equalization): Enhances epiphyseal-metaphyseal growth plate contrast.
4. Dual-Branch Resolution Alignment: High-res for ConvNeXt micro-morphology, window-aligned for Swin macro-structure.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from typing import Tuple, Dict, Any, Optional


class LetterboxSquarePad:
    """Pads an image to a square aspect ratio without geometric distortion."""
    def __init__(self, target_size: int = 512, pad_value: int = 0):
        self.target_size = target_size
        self.pad_value = pad_value

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h == w and h == self.target_size:
            return img

        # Compute scaling factor preserving aspect ratio
        scale = self.target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Compute symmetric padding
        pad_top = (self.target_size - new_h) // 2
        pad_bottom = self.target_size - new_h - pad_top
        pad_left = (self.target_size - new_w) // 2
        pad_right = self.target_size - new_w - pad_left

        if len(img.shape) == 2:
            padded = cv2.copyMakeBorder(
                resized, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=self.pad_value
            )
        else:
            padded = cv2.copyMakeBorder(
                resized, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[self.pad_value] * img.shape[2]
            )

        return padded


class HandOrientationNormalizer:
    """Normalizes hand radiographs to canonical left-hand atlas convention."""
    @staticmethod
    def is_right_hand_heuristic(img_gray: np.ndarray) -> bool:
        """Heuristic check based on center of mass and horizontal thumb asymmetry.
        
        In standard pronated left-hand PA radiographs, the thumb (first ray) is on the right side.
        In right-hand radiographs, the thumb is on the left side.
        """
        h, w = img_gray.shape
        # Inspect middle vertical third to locate thumb divergence
        mid_strip = img_gray[int(h * 0.3):int(h * 0.7), :]
        left_half_density = np.sum(mid_strip[:, :w // 2] > 30)
        right_half_density = np.sum(mid_strip[:, w // 2:] > 30)
        
        # If left half significantly dominates divergence, it suggests right hand
        return left_half_density > (right_half_density * 1.35)

    @staticmethod
    def normalize_to_left_hand(img: np.ndarray, is_right_hand: Optional[bool] = None) -> np.ndarray:
        """Flips right-hand image horizontally to mimic standard left-hand radiograph."""
        if is_right_hand is None:
            gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            is_right_hand = HandOrientationNormalizer.is_right_hand_heuristic(gray)

        if is_right_hand:
            return cv2.flip(img, 1)  # Horizontal flip
        return img


class CLAHEContrastEnhancer:
    """Applies Contrast Limited Adaptive Histogram Equalization to accentuate cartilage and bone boundaries."""
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img_gray: np.ndarray) -> np.ndarray:
        return self.clahe.apply(img_gray)


class DualBranchTransform:
    """Constructs coordinated transforms for ConvNeXt (micro-morphology) and Swin (macro-context)."""
    def __init__(
        self,
        conv_size: int = 384,
        swin_size: int = 224,
        is_training: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        self.conv_size = conv_size
        self.swin_size = swin_size
        self.is_training = is_training
        self.mean = mean
        self.std = std
        self.letterbox = LetterboxSquarePad(target_size=512)
        self.clahe = CLAHEContrastEnhancer(clip_limit=2.0, tile_grid_size=(8, 8))

    def __call__(self, img_rgb: np.ndarray, is_right_hand: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes RGB input and returns (img_conv, img_swin) normalized tensors."""
        # 1. Orientation normalization to canonical left-hand
        norm_img = HandOrientationNormalizer.normalize_to_left_hand(img_rgb, is_right_hand)

        # 2. Letterbox to preserve geometry
        letterboxed = self.letterbox(norm_img)

        # 3. CLAHE enhancement on luminance / gray channel
        gray = cv2.cvtColor(letterboxed, cv2.COLOR_RGB2GRAY)
        clahe_gray = self.clahe(gray)
        enhanced_rgb = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2RGB)

        # 4. Subtle scientific data augmentations (only if training)
        if self.is_training:
            # Gentle rotation [-7 to +7 deg] representing patient wrist posture variation
            angle = np.random.uniform(-7.0, 7.0)
            h, w = enhanced_rgb.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            enhanced_rgb = cv2.warpAffine(enhanced_rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # 5. Dual resolution preparation
        conv_resized = cv2.resize(enhanced_rgb, (self.conv_size, self.conv_size), interpolation=cv2.INTER_LINEAR)
        swin_resized = cv2.resize(enhanced_rgb, (self.swin_size, self.swin_size), interpolation=cv2.INTER_LINEAR)

        # 6. Tensor conversion and ImageNet normalization
        conv_tensor = torch.from_numpy(conv_resized).permute(2, 0, 1).float() / 255.0
        swin_tensor = torch.from_numpy(swin_resized).permute(2, 0, 1).float() / 255.0

        for c in range(3):
            conv_tensor[c] = (conv_tensor[c] - self.mean[c]) / self.std[c]
            swin_tensor[c] = (swin_tensor[c] - self.mean[c]) / self.std[c]

        return conv_tensor, swin_tensor


class AnatomicalCarpalCrop:
    """Extracts high-resolution anatomical zoom on Wrist / Carpal cluster & Distal Epiphysis.

    In standardized pediatric PA hand radiographs, the carpals (Capitate, Hamate, Triquetral, etc.)
    and the distal radial/ulnar epiphyseal plates reside in the lower-central segment of the hand.
    This crop preserves sub-millimeter trabecular bone micro-patterns without extreme downsampling blur.
    """
    def __init__(
        self,
        crop_box: Tuple[float, float, float, float] = (0.45, 0.95, 0.15, 0.85),
        target_size: int = 256
    ):
        self.ymin, self.ymax, self.xmin, self.xmax = crop_box
        self.target_size = target_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        y1 = max(0, int(h * self.ymin))
        y2 = min(h, int(h * self.ymax))
        x1 = max(0, int(w * self.xmin))
        x2 = min(w, int(w * self.xmax))

        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            crop = img
        return cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)


class MultiCropDualBranchTransform:
    """Tri-stream or Dual-crop transform providing both Whole-Hand context and High-Res Wrist ROI."""
    def __init__(
        self,
        hand_size: int = 384,
        roi_size: int = 256,
        is_training: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        self.hand_size = hand_size
        self.roi_size = roi_size
        self.is_training = is_training
        self.mean = mean
        self.std = std
        self.letterbox = LetterboxSquarePad(target_size=512)
        self.clahe = CLAHEContrastEnhancer(clip_limit=2.0, tile_grid_size=(8, 8))
        self.carpal_cropper = AnatomicalCarpalCrop(crop_box=(0.45, 0.95, 0.15, 0.85), target_size=roi_size)

    def __call__(self, img_rgb: np.ndarray, is_right_hand: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_img = HandOrientationNormalizer.normalize_to_left_hand(img_rgb, is_right_hand)
        letterboxed = self.letterbox(norm_img)

        gray = cv2.cvtColor(letterboxed, cv2.COLOR_RGB2GRAY)
        clahe_gray = self.clahe(gray)
        enhanced_rgb = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2RGB)

        # High-res optical zoom before aggressive global downscaling
        roi_crop = self.carpal_cropper(enhanced_rgb)

        if self.is_training:
            angle = np.random.uniform(-6.0, 6.0)
            h, w = enhanced_rgb.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            enhanced_rgb = cv2.warpAffine(enhanced_rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        hand_resized = cv2.resize(enhanced_rgb, (self.hand_size, self.hand_size), interpolation=cv2.INTER_LINEAR)

        t_hand = torch.from_numpy(hand_resized).permute(2, 0, 1).float() / 255.0
        t_roi = torch.from_numpy(roi_crop).permute(2, 0, 1).float() / 255.0

        for c in range(3):
            t_hand[c] = (t_hand[c] - self.mean[c]) / self.std[c]
            t_roi[c] = (t_roi[c] - self.mean[c]) / self.std[c]

        return t_hand, t_roi

