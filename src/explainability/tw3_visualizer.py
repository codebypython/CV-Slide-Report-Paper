"""TW3 (Tanner-Whitehouse 3) Anatomical Grounding and 4-Panel Clinical Explainability Visualizer.

Correlates neural attention patterns against the 13 gold-standard TW3 skeletal maturity ROIs:
1. Distal Radius (Growth plate & styloid process)
2. Distal Ulna (Epiphyseal fusion)
3. Metacarpal I (Thumb ray metacarpal)
4. Metacarpal III (Middle ray metacarpal)
5. Metacarpal V (Little ray metacarpal)
6. Proximal Phalanx I
7. Proximal Phalanx III
8. Proximal Phalanx V
9. Middle Phalanx III
10. Middle Phalanx V
11. Distal Phalanx I
12. Distal Phalanx III
13. Distal Phalanx V
+ Carpal cluster (Capitate, Hamate, Triquetrum, Lunate, Scaphoid, Trapezium, Trapezoid, Pisiform).
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any


# Standard normalized canonical coordinates for 13 TW3 anatomical keypoints on 512x512 canvas
TW3_KEYPOINTS_CANONICAL = {
    "Radius (Distal)": (210, 400, 24),
    "Ulna (Distal)": (290, 415, 20),
    "Carpal Cluster": (250, 360, 45),
    "Metacarpal I": (185, 300, 18),
    "Metacarpal III": (255, 245, 18),
    "Metacarpal V": (320, 275, 18),
    "Proximal Phalanx I": (160, 245, 16),
    "Proximal Phalanx III": (255, 160, 16),
    "Proximal Phalanx V": (335, 210, 16),
    "Middle Phalanx III": (255, 115, 14),
    "Middle Phalanx V": (340, 175, 14),
    "Distal Phalanx I": (145, 205, 14),
    "Distal Phalanx III": (255, 75, 14),
    "Distal Phalanx V": (345, 145, 14)
}


def compute_tw3_attention_concentration(
    heatmap: np.ndarray,
    keypoints: Dict[str, Tuple[int, int, int]] = None
) -> float:
    """Computes the ratio of saliency energy localized within legitimate TW3 regions vs background."""
    if keypoints is None:
        keypoints = TW3_KEYPOINTS_CANONICAL

    h, w = heatmap.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.float32)

    for name, (cx, cy, radius) in keypoints.items():
        cv2.circle(roi_mask, (cx, cy), radius, 1.0, -1)

    total_energy = float(np.sum(heatmap)) + 1e-8
    tw3_energy = float(np.sum(heatmap * roi_mask))

    concentration_ratio = tw3_energy / total_energy
    return concentration_ratio


def generate_clinical_explainability_figure(
    img_raw: np.ndarray,
    gradcam_map: np.ndarray,
    attention_map: np.ndarray,
    sample_id: Any,
    true_age: float,
    pred_age: float,
    is_male: bool,
    output_path: str
) -> Dict[str, float]:
    """Renders and saves a 4-panel clinical validation figure formatted for scientific publication."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Ensure grayscale base is converted to RGB
    if len(img_raw.shape) == 2:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_raw.copy()

    # Colormaps for heatmaps
    cam_color = cv2.applyColorMap(np.uint8(255 * gradcam_map), cv2.COLORMAP_JET)
    cam_overlay = cv2.addWeighted(img_rgb, 0.55, cam_color, 0.45, 0)

    attn_color = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_MAGMA)
    attn_overlay = cv2.addWeighted(img_rgb, 0.55, attn_color, 0.45, 0)

    # Panel 4: TW3 13-Region Ground-Truth Mapping
    tw3_overlay = img_rgb.copy()
    for name, (cx, cy, radius) in TW3_KEYPOINTS_CANONICAL.items():
        cv2.circle(tw3_overlay, (cx, cy), radius, (0, 255, 0), 2)
        cv2.circle(tw3_overlay, (cx, cy), 3, (0, 0, 255), -1)

    # Concentration scores
    cam_tw3_ratio = compute_tw3_attention_concentration(gradcam_map)
    attn_tw3_ratio = compute_tw3_attention_concentration(attention_map)

    # Plot 4 panels
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), dpi=300)

    # Panel 1: Original Radiograph
    sex_str = "Male" if is_male else "Female"
    axes[0].imshow(img_rgb)
    axes[0].set_title(
        f"(A) Radiograph #{sample_id}\nTarget: {true_age:.1f}m | Pred: {pred_age:.1f}m ({sex_str})",
        fontsize=11, fontweight="bold"
    )
    axes[0].axis("off")

    # Panel 2: ConvNeXt Grad-CAM
    axes[1].imshow(cam_overlay)
    axes[1].set_title(
        f"(B) ConvNeXt Grad-CAM\nMicro-Morphology (TW3 Ratio: {cam_tw3_ratio*100:.1f}%)",
        fontsize=11, fontweight="bold"
    )
    axes[1].axis("off")

    # Panel 3: Swin Cross-Attention
    axes[2].imshow(attn_overlay)
    axes[2].set_title(
        f"(C) Swin Cross-Attention\nMacro-Context (TW3 Ratio: {attn_tw3_ratio*100:.1f}%)",
        fontsize=11, fontweight="bold"
    )
    axes[2].axis("off")

    # Panel 4: TW3 13 Anatomical ROIs
    axes[3].imshow(tw3_overlay)
    axes[3].set_title(
        "(D) TW3 Gold-Standard ROIs\n(Radius, Ulna, Phalanges, Carpals)",
        fontsize=11, fontweight="bold"
    )
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "gradcam_tw3_ratio": cam_tw3_ratio,
        "attention_tw3_ratio": attn_tw3_ratio
    }
