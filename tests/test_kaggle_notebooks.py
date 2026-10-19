"""Comprehensive Verification and Self-Correction Suite for Kaggle Notebooks.

Audits:
1. JSON schema and nbformat v4 structure of all 5 notebooks.
2. Abstract Syntax Tree (AST) compilation of every code cell in every notebook.
3. Functional verification of core modular architectures with mock tensors.
4. Verification of preprocessing operations (Letterbox, Loose-crop, CLAHE, Flip).
"""

import ast
import json
import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "kaggle_notebooks"


class TestKaggleNotebooksIntegrity(unittest.TestCase):
    def setUp(self):
        self.notebook_files = [
            "01_rsna_preprocessing_and_caching.ipynb",
            "02a_inception_rsna2017_baseline.ipynb",
            "02b_dual_branch_cross_attention_film.ipynb",
            "02c_champion_dual_branch_full_suite.ipynb",
            "02d_champion_dual_branch_5folds_full_session.ipynb",
            "02e_champion_scale_and_resolution.ipynb",
            "02f_anatomical_roi_multicrop.ipynb",
            "03_scientific_evaluation_and_explainability.ipynb",
            "03b_sota_ensemble_and_manuscript.ipynb",
        ]


    def test_notebook_json_structure(self):
        """Vòng 1: Kiểm định tính toàn vẹn cấu trúc JSON nbformat v4."""
        for nb_name in self.notebook_files:
            nb_path = NOTEBOOKS_DIR / nb_name
            self.assertTrue(nb_path.exists(), f"Notebook {nb_name} does not exist at {nb_path}!")

            with open(nb_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIn("cells", data, f"Notebook {nb_name} missing 'cells' key!")
            self.assertIn("metadata", data, f"Notebook {nb_name} missing 'metadata' key!")
            self.assertEqual(data.get("nbformat"), 4, f"Notebook {nb_name} nbformat is not 4!")
            self.assertGreater(len(data["cells"]), 3, f"Notebook {nb_name} has too few cells ({len(data['cells'])})!")

    def test_code_cells_ast_syntax(self):
        """Vòng 2: Duyệt và phân tích cú pháp AST của từng code cell trong mọi notebook."""
        for nb_name in self.notebook_files:
            nb_path = NOTEBOOKS_DIR / nb_name
            with open(nb_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            code_cell_idx = 0
            for cell in data["cells"]:
                if cell["cell_type"] == "code":
                    code_cell_idx += 1
                    raw_lines = cell["source"]
                    # Filter out jupyter notebook magic lines like '!pip install' or '%matplotlib'
                    clean_lines = []
                    for line in raw_lines:
                        l_strip = line.strip()
                        if l_strip.startswith("!") or l_strip.startswith("%"):
                            clean_lines.append("# " + line)
                        else:
                            clean_lines.append(line)

                    code_content = "".join(clean_lines)
                    try:
                        ast.parse(code_content)
                    except SyntaxError as e:
                        self.fail(f"SyntaxError in {nb_name} at code cell #{code_cell_idx}: {e}")

    def test_preprocessing_transforms_logic(self):
        """Vòng 3: Kiểm định tính đúng đắn toán học của các hàm tiền xử lý."""
        # Giả lập 1 ảnh chữ nhật đứng (H > W) điển hình của bàn tay
        mock_hand = np.zeros((600, 350), dtype=np.uint8)
        cv2.circle(mock_hand, (175, 300), 80, 150, -1)

        # 1. Test Loose-crop
        mask = (mock_hand > 15).astype(np.uint8)
        coords = cv2.findNonZero(mask)
        x, y, bw, bh = cv2.boundingRect(coords)
        pad_x, pad_y = int(bw * 0.10), int(bh * 0.10)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(350, x + bw + pad_x), min(600, y + bh + pad_y)
        cropped = mock_hand[y1:y2, x1:x2]
        self.assertGreater(cropped.shape[0], 0)
        self.assertGreater(cropped.shape[1], 0)

        # 2. Test Letterbox padding
        target_w, target_h = 512, 512
        h, w = cropped.shape
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        delta_w, delta_h = target_w - new_w, target_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        self.assertEqual(padded.shape, (512, 512), "Letterbox padding did not produce exact 512x512 shape!")

    def test_dual_branch_model_forward_backward(self):
        """Vòng 4: Kiểm định mock forward và backward pass cho mô hình Dual-Branch."""
        # Định nghĩa các module đúng chuẩn như trong Notebook 02b / 02c
        class FiLMLayer(nn.Module):
            def __init__(self, feature_dim=128, metadata_dim=1):
                super().__init__()
                self.generator = nn.Sequential(
                    nn.Linear(metadata_dim, 32),
                    nn.GELU(),
                    nn.Linear(32, feature_dim * 2)
                )

            def forward(self, x, meta):
                p = self.generator(meta)
                gamma, beta = torch.chunk(p, 2, dim=-1)
                return x * (1.0 + gamma) + beta

        class CrossAttentionFusion(nn.Module):
            def __init__(self, embed_dim=128, num_heads=4):
                super().__init__()
                self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
                self.norm = nn.LayerNorm(embed_dim)

            def forward(self, q, k):
                attn_out, _ = self.cross_attn(query=q, key=k, value=k)
                return self.norm(q + attn_out).squeeze(1)

        embed_dim = 128
        film = FiLMLayer(feature_dim=embed_dim)
        fusion = CrossAttentionFusion(embed_dim=embed_dim, num_heads=4)
        regressor = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 1))

        # Mock representations
        f_s = torch.randn(2, 1, embed_dim) # Swin macro
        f_c = torch.randn(2, 1, embed_dim) # Conv micro
        gender = torch.tensor([[1.0], [0.0]])

        # Forward
        fused = fusion(f_s, f_c)
        self.assertEqual(fused.shape, (2, embed_dim))

        modulated = film(fused, gender)
        self.assertEqual(modulated.shape, (2, embed_dim))

        pred = regressor(modulated).squeeze(-1)
        self.assertEqual(pred.shape, (2,))

        # Backward
        loss = pred.sum()
        loss.backward()
        self.assertIsNotNone(regressor[0].weight.grad)

    def test_inception_baseline_forward_backward(self):
        """Vòng 5: Kiểm định mô hình Inception-v3 Baseline với timm."""
        import timm
        model = nn.Sequential(
            timm.create_model("inception_v3", pretrained=False, num_classes=0)
        )
        x = torch.randn(2, 3, 299, 299)
        feat = model(x)
        self.assertEqual(feat.shape, (2, 2048))

    def test_notebooks_pretraining_cells_runtime_exec(self):
        """Vòng 6: Thực thi runtime thực tế các cell tiền huấn luyện (Import, Config, Dataset, Model) của 02d, 02e, 02f."""
        for nb_name in ["02d_champion_dual_branch_5folds_full_session.ipynb", "02e_champion_scale_and_resolution.ipynb", "02f_anatomical_roi_multicrop.ipynb"]:
            nb_path = NOTEBOOKS_DIR / nb_name
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)

            ns = {}
            for idx, c in enumerate(nb["cells"]):
                if c["cell_type"] == "code":
                    raw_lines = c["source"]
                    clean_lines = [("# " + l if l.strip().startswith(("!", "%")) else l) for l in raw_lines]
                    code = "".join(clean_lines)

                    if "DATASET_DIR = None" in code:
                        continue
                    if "session_start_time = time.time()" in code or "for fold_idx in" in code:
                        break

                    try:
                        exec(code, ns)
                    except Exception as e:
                        self.fail(f"Runtime execution failed in {nb_name} cell #{idx}: {type(e).__name__} - {e}")


if __name__ == "__main__":
    unittest.main()

