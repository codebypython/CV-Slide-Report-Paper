"""PyTorch Dataset module for RSNA Pediatric Bone Age radiographs."""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict, Any, Union

from .transforms import DualBranchTransform


class RSNABoneAgeDataset(Dataset):
    """Standardized dataset for RSNA bone age assessment with dual-branch preprocessing."""
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        transform: Optional[DualBranchTransform] = None,
        is_training: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.is_training = is_training
        self.transform = transform or DualBranchTransform(is_training=is_training)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        sample_id = row["id"]
        
        # Support both int id (e.g. 1377 -> 1377.png) and string id
        img_filename = f"{sample_id}.png" if not str(sample_id).endswith((".png", ".jpg")) else str(sample_id)
        img_path = os.path.join(self.img_dir, img_filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Bone age radiograph not found at {img_path}")

        # Read as grayscale first
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise ValueError(f"Failed to decode image at {img_path}")

        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        # Apply dual-branch transform
        img_conv, img_swin = self.transform(img_rgb)

        # Parse gender metadata (1.0 for male, 0.0 for female)
        male_val = row["male"]
        if isinstance(male_val, (bool, np.bool_)):
            gender_scalar = 1.0 if male_val else 0.0
        elif isinstance(male_val, str):
            gender_scalar = 1.0 if male_val.strip().lower() in ["1", "true", "male", "m"] else 0.0
        else:
            gender_scalar = float(male_val)

        gender = torch.tensor([gender_scalar], dtype=torch.float32)
        bone_age = torch.tensor(float(row["boneage"]), dtype=torch.float32)

        meta = {
            "id": sample_id,
            "male": gender_scalar == 1.0,
            "bone_age": float(row["boneage"]),
            "age_bracket": row.get("age_bracket", "Unknown")
        }

        return img_conv, img_swin, gender, bone_age, meta
