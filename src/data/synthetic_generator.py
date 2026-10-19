"""Synthetic pediatric hand radiograph generator for test suite, CI validation, and pipeline verification.

Generates realistic phantom radiographs with:
- Hand silhouette, palm arch, and wrist base.
- Five distinct rays (thumb, index, long, ring, small fingers) with phalanges and growth plates.
- Carpal bone ossification centers with density scaling proportional to synthetic bone age.
- Realistic label distributions (age: 1 to 228 months, sex: 50% male, 50% female).
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def draw_synthetic_hand_xray(bone_age_months: float, is_male: bool, img_size: int = 512) -> np.ndarray:
    """Renders a synthetic hand radiograph phantom simulating ossification changes across age."""
    canvas = np.zeros((img_size, img_size), dtype=np.uint8)

    # Base background noise mimicking X-ray scatter
    noise = np.random.normal(12, 3, (img_size, img_size)).clip(0, 30).astype(np.uint8)
    canvas = cv2.add(canvas, noise)

    maturity_ratio = min(max(bone_age_months / 216.0, 0.05), 1.0)
    bone_intensity = int(140 + 100 * maturity_ratio)
    epiphysis_gap = int(max(1, int(10 * (1.0 - maturity_ratio))))

    # 1. Soft tissue palm and wrist silhouette
    cv2.ellipse(canvas, (256, 360), (140, 110), 0, 0, 360, 45, -1)
    cv2.rectangle(canvas, (170, 420), (342, 512), 40, -1)

    # 2. Forearm bones (Radius and Ulna)
    # Radius (lateral/thumb side)
    cv2.line(canvas, (210, 512), (210, 410), bone_intensity, 24)
    # Distal radial epiphysis
    cv2.ellipse(canvas, (210, 400 - epiphysis_gap), (20, 8), 0, 0, 360, bone_intensity + 15, -1)

    # Ulna (medial side)
    cv2.line(canvas, (290, 512), (290, 425), int(bone_intensity * 0.9), 18)
    # Distal ulnar epiphysis
    cv2.ellipse(canvas, (290, 415 - epiphysis_gap), (12, 6), 0, 0, 360, bone_intensity + 10, -1)

    # 3. Carpal bones (Ossification centers emerge progressively)
    # Young children have only 2 centers (capitate, hamate), older children have all 8
    num_carpals = min(8, max(1, int(maturity_ratio * 8.5)))
    carpal_centers = [
        (250, 360),  # Capitate
        (275, 365),  # Hamate
        (230, 375),  # Triquetrum
        (255, 390),  # Lunate
        (225, 350),  # Scaphoid
        (205, 335),  # Trapezium
        (225, 325),  # Trapezoid
        (295, 385),  # Pisiform
    ]
    for i in range(num_carpals):
        cx, cy = carpal_centers[i]
        radius = int(5 + 7 * maturity_ratio)
        cv2.circle(canvas, (cx, cy), radius, bone_intensity - 10, -1)

    # 4. Metacarpals and Phalanges (5 rays)
    finger_params = [
        # (metacarpal_base, metacarpal_tip, phalanx_lengths, angle_deg, thickness)
        ((200, 330), (170, 270), [28, 24], -25, 14),   # Ray 1: Thumb
        ((225, 310), (210, 215), [36, 26, 18], -7, 12),# Ray 2: Index
        ((255, 300), (255, 190), [42, 30, 20], 0, 13), # Ray 3: Middle
        ((285, 305), (295, 205), [38, 28, 19], 5, 12), # Ray 4: Ring
        ((310, 320), (330, 235), [30, 20, 15], 12, 10) # Ray 5: Little
    ]

    for base, tip, phalanges, angle, thick in finger_params:
        # Metacarpal shaft
        cv2.line(canvas, base, tip, bone_intensity, thick)
        curr_pt = np.array(tip, dtype=np.float32)
        rad = np.radians(90 + angle)
        dir_vec = np.array([-np.cos(rad), -np.sin(rad)], dtype=np.float32)

        for plen in phalanges:
            # Epiphyseal growth plate gap
            epiphysis_pt = curr_pt + dir_vec * 4
            phalanx_start = curr_pt + dir_vec * (4 + epiphysis_gap)
            phalanx_end = phalanx_start + dir_vec * plen

            # Small epiphysis disk
            cv2.circle(canvas, (int(epiphysis_pt[0]), int(epiphysis_pt[1])), max(2, thick // 2), bone_intensity, -1)
            # Phalanx shaft
            cv2.line(
                canvas,
                (int(phalanx_start[0]), int(phalanx_start[1])),
                (int(phalanx_end[0]), int(phalanx_end[1])),
                bone_intensity - 15,
                max(4, thick - 3)
            )
            curr_pt = phalanx_end

    # Gaussian blur to soften edges into realistic radiograph radiopacity
    canvas = cv2.GaussianBlur(canvas, (5, 5), 1.2)
    return canvas


def generate_synthetic_dataset(
    output_dir: str,
    num_samples: int = 100,
    seed: int = 42
) -> Tuple[str, str]:
    """Generates a complete synthetic RSNA-style dataset directory with CSV and PNG images."""
    np.random.seed(seed)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    records = []
    # Sample ages realistically from standard RSNA age distribution (mean ~ 127 mo, std ~ 41)
    ages = np.random.normal(loc=125, scale=42, size=num_samples).clip(6, 228)
    males = np.random.choice([True, False], size=num_samples)

    for idx in range(num_samples):
        sample_id = 1000 + idx
        age = float(round(ages[idx], 1))
        is_male = bool(males[idx])

        # Render synthetic X-ray
        img = draw_synthetic_hand_xray(age, is_male, img_size=512)
        img_filename = f"{sample_id}.png"
        img_path = os.path.join(images_dir, img_filename)
        cv2.imwrite(img_path, img)

        records.append({
            "id": sample_id,
            "boneage": age,
            "male": is_male
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "boneage_training_dataset.csv")
    df.to_csv(csv_path, index=False)
    return csv_path, images_dir
