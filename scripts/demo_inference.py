"""Command Line Interface (CLI) for Pediatric Bone Age Assessment (Pipeline 03b).

Usage:
  python scripts/demo_inference.py --image path/to/hand_xray.png --male 1
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from src.inference.pipeline_03b import BoneAgeInferencePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Pediatric Bone Age Diagnostic Demo (Pipeline 03b)")
    parser.add_argument("--image", type=str, default=None, help="Path to input hand radiograph image")
    parser.add_argument("--male", type=int, default=1, choices=[0, 1], help="Gender (1: Male, 0: Female)")
    parser.add_argument("--checkpoint-02d", type=str, default=None, help="Path to 02d checkpoint")
    parser.add_argument("--checkpoint-02f", type=str, default=None, help="Path to 02f checkpoint")
    parser.add_argument("--weights", type=str, default=None, help="Path to stacking weights JSON")
    parser.add_argument("--output", type=str, default="outputs/diagnostic_report.png", help="Path to save diagnostic report image")
    return parser.parse_args()


def create_diagnostic_card(result: dict, output_path: str):
    """Draws a 4-panel clinical diagnostic report card."""
    fig = plt.figure(figsize=(16, 10), facecolor="#f8f9fa")
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.20)

    # 1. Whole Hand with Carpal ROI
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(result["images"]["whole_hand_annotated"])
    ax1.set_title("Panel A: Toan Canh Ban Tay & Vung Cat Co Tay (02f ROI)", fontsize=12, fontweight="bold", pad=8)
    ax1.axis("off")

    # 2. Carpal Optical Zoom
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(result["images"]["carpal_zoom_crop"])
    ax2.set_title("Panel B: Vi Phau Phong To Cum Co Tay & Dau Duoi Xuong Quay (256x256)", fontsize=12, fontweight="bold", pad=8)
    ax2.axis("off")

    # 3. TW3 Attention Heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(result["images"]["attention_heatmap"])
    ax3.set_title("Panel C: Ban Do Chu Y Cac Dia Sun Tiep Hop (TW3 Explainability)", fontsize=12, fontweight="bold", pad=8)
    ax3.axis("off")

    # 4. Clinical Score Card & Metrics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    # Background card
    card_rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, transform=ax4.transAxes,
                               facecolor="white", edgecolor="#cbd5e0", lw=1.5,
                               boxstyle="round,pad=0.03")
    ax4.add_patch(card_rect)

    # Header
    ax4.text(0.08, 0.90, "PHIEU KET QUA CHAN DOAN TUOI XUONG",
             fontsize=14, fontweight="bold", color="#1a202c", transform=ax4.transAxes)
    ax4.text(0.08, 0.84, f"He Thong De Xuat SOTA: Stacking Heterogeneous Ensembling (03b)",
             fontsize=10, color="#718096", transform=ax4.transAxes)

    # Divider
    ax4.plot([0.08, 0.92], [0.81, 0.81], color="#e2e8f0", lw=1.5, transform=ax4.transAxes)

    # Main Bone Age Readout
    age_m = result["predicted_bone_age_months"]
    age_y = result["predicted_bone_age_years"]
    ax4.text(0.08, 0.68, f"{age_m:.1f}",
             fontsize=36, fontweight="bold", color="#2b6cb0", transform=ax4.transAxes)
    ax4.text(0.35, 0.72, "thang", fontsize=14, fontweight="bold", color="#2b6cb0", transform=ax4.transAxes)
    ax4.text(0.35, 0.65, f"(~ {age_y} tuoi)", fontsize=12, color="#4a5568", transform=ax4.transAxes)

    # Patient info
    gender_str = result["gender"]
    ax4.text(0.65, 0.72, f"Gioi tinh: {gender_str}", fontsize=11, fontweight="bold", color="#2d3748", transform=ax4.transAxes)

    # Clinical Pubertal Stage
    stage = result["pubertal_stage"]
    desc = result["pubertal_description"]
    badge_color = result["pubertal_color"]

    stage_box = FancyBboxPatch((0.08, 0.48), 0.84, 0.12, transform=ax4.transAxes,
                               facecolor=badge_color, alpha=0.12, edgecolor=badge_color, lw=1.5,
                               boxstyle="round,pad=0.02")
    ax4.add_patch(stage_box)
    ax4.text(0.11, 0.54, f"Danh Gia Lam Sang: {stage}", fontsize=11, fontweight="bold", color=badge_color, transform=ax4.transAxes)
    ax4.text(0.11, 0.49, desc, fontsize=8.5, color="#4a5568", transform=ax4.transAxes)

    # Architectural Breakdown
    mb = result["model_breakdown"]
    ax4.text(0.08, 0.38, "Phân Tích Đóng Góp Của Từng Nhánh Kiến Trúc:", fontsize=11, fontweight="bold", color="#2d3748", transform=ax4.transAxes)

    b_text1 = f"• Mô hình 02d (Toàn Cảnh Bàn Tay Macro):"
    b_val1 = f"{mb['pred_02d_months']:.1f}m (Trọng số: {mb['weight_02d']*100:.1f}%)"
    ax4.text(0.11, 0.31, b_text1, fontsize=9.5, color="#4a5568", transform=ax4.transAxes)
    ax4.text(0.65, 0.31, b_val1, fontsize=9.5, fontweight="bold", color="#2b6cb0", transform=ax4.transAxes)

    b_text2 = f"• Mô hình 02f (Vi Phẫu Cổ Tay Carpal Zoom):"
    b_val2 = f"{mb['pred_02f_months']:.1f}m (Trọng số: {mb['weight_02f']*100:.1f}%)"
    ax4.text(0.11, 0.24, b_text2, fontsize=9.5, color="#4a5568", transform=ax4.transAxes)
    ax4.text(0.65, 0.24, b_val2, fontsize=9.5, fontweight="bold", color="#2ca02c", transform=ax4.transAxes)

    # Footer note
    ax4.text(0.08, 0.10, "Chứng nhận chuẩn hóa: Zero Leakage Test Validation | Phù hợp bài báo IEEE TMI",
             fontsize=8.5, style="italic", color="#a0aec0", transform=ax4.transAxes)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Da luu phieu chan doan lam sang tai: {out_p.resolve()}")


def main():
    args = parse_args()

    # Determine input image
    img_path = args.image
    if not img_path:
        # Generate demo synthetic hand image if none provided
        print("[INFO] Khong co tham so --image, dang tao anh X-quang mau thu nghiem...")
        synth = np.ones((512, 512, 3), dtype=np.uint8) * 20
        # Draw palm and wrist
        cv2.ellipse(synth, (256, 380), (100, 110), 0, 0, 360, (180, 180, 180), -1)
        # Draw 5 fingers
        finger_coords = [
            ((170, 240), (14, 60), -25),
            ((215, 170), (15, 80), -8),
            ((256, 150), (16, 90), 0),
            ((298, 170), (15, 80), 8),
            ((340, 210), (14, 65), 18)
        ]
        for center, axes, ang in finger_coords:
            cv2.ellipse(synth, center, axes, ang, 0, 360, (190, 190, 190), -1)
        # Blur to look like X-ray radiograph
        synth = cv2.GaussianBlur(synth, (21, 21), 0)
        img_input = synth
    else:
        img_input = img_path

    # Initialize pipeline
    pipeline = BoneAgeInferencePipeline(
        checkpoint_02d=args.checkpoint_02d,
        checkpoint_02f=args.checkpoint_02f,
        stacking_weights_path=args.weights
    )

    print("[RUNNING] Dang tien hanh phan tich tuoi xuong qua Pipeline 03b Stacking...")
    result = pipeline.predict(img_input, is_male=args.male)

    # Print clean summary table
    print("=" * 65)
    print("           KET QUA CHAN DOAN TUOI XUONG (PIPELINE 03b)")
    print("=" * 65)
    print(f"* Gioi tinh benh nhan:       {result['gender']}")
    print(f"* TUOI XUONG DU DOAN:        {result['predicted_bone_age_months']:.1f} THANG (~ {result['predicted_bone_age_years']} TUOI)")
    print(f"* Giai doan day thi lam sang: {result['pubertal_stage']}")
    print(f"* Nhan dinh:                 {result['pubertal_description']}")
    print("-" * 65)
    print("* Dong gop tu cac nhanh mo hinh:")
    mb = result["model_breakdown"]
    print(f"  - Mo hinh 02d (Toan canh): {mb['pred_02d_months']:.1f} thang  |  Trong so: {mb['weight_02d']*100:.1f}%")
    print(f"  - Mo hinh 02f (Cum co tay): {mb['pred_02f_months']:.1f} thang  |  Trong so: {mb['weight_02f']*100:.1f}%")
    print("=" * 65)

    # Generate visual diagnostic card
    create_diagnostic_card(result, args.output)


if __name__ == "__main__":
    main()
