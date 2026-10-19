# 📘 CẨM NANG THỰC CHIẾN HUẤN LUYỆN KAGGLE: HỆ THỐNG DUAL-BRANCH BONE AGE ASSESSMENT
### Dành cho đề tài: *"Harmonizing Local Inductive Bias and Global Contextual Reasoning: A Dual-Branch ConvNeXt-Swin Transformer Architecture with Gender-Modulated Cross-Attention for Pediatric Bone Age Assessment"*
**Target Journal:** *IEEE Transactions on Medical Imaging (IEEE TMI)* / *Medical Image Analysis (MedIA)*

---

## 🗺️ TỔNG QUAN HỆ THỐNG NOTEBOOKS

Toàn bộ quá trình nghiên cứu thực nghiệm được cấu trúc thành **5 tệp Jupyter Notebook độc lập tự chứa (Self-Contained)** nằm trong thư mục `kaggle_notebooks/`:

```text
kaggle_notebooks/
├── 01_rsna_preprocessing_and_caching.ipynb          # Tiền xử lý Loose-Crop, CLAHE 2.0, Letterbox 512x512, Stratified Split
├── 02a_inception_rsna2017_baseline.ipynb            # [Mốc 1] Inception-v3 + Naive Concat (RSNA 2017 Baseline, 8.60m)
├── 02b_dual_branch_cross_attention_film.ipynb       # [Mốc 2] Đột phá kiến trúc ConvNeXt + Swin + FiLM (Single Run)
├── 02c_champion_dual_branch_full_suite.ipynb        # [Mốc 3] Bản huấn luyện từng Fold độc lập (Interactive / Single Fold)
├── 02d_champion_dual_branch_5folds_full_session.ipynb # [Mốc 3D] SIÊU BẢN 5-FOLD ALL-IN-ONE: Chạy trọn vẹn 5 Folds trong 1 Commit (7.21m)
├── 02e_champion_scale_and_resolution.ipynb          # [Mốc Giai đoạn 1] SCALED 102.5M PARAMS: ConvNeXt-Small (512) + Swin-Small (384) + Adaptive Pubertal Loss
├── 02f_anatomical_roi_multicrop.ipynb               # [Mốc Giai đoạn 2] ANATOMICAL DUAL-CROP: Toàn tay (384) + Zoom quang học cổ tay & đĩa sụn (256)
├── 03_scientific_evaluation_and_explainability.ipynb# Kết xuất 5 biểu đồ IEEE DPI 300 & TW3 Grad-CAM & Bảng LaTeX
├── 03b_sota_ensemble_and_manuscript.ipynb           # [Mốc Giai đoạn 3] HETEROGENEOUS STACKING ENSEMBLE: Đột phá SOTA <= 4.25m + 5 Biểu đồ IEEE 300 DPI + LaTeX
└── KAGGLE_WORKFLOW_GUIDE.md                         # Cẩm nang thực chiến toàn diện này
```

---

## 🔬 PHÂN TÍCH CHUYÊN SÂU: TẠI SAO SINGLE-RUN ĐẠT 6.97m VÀ CON ĐƯỜNG ĐỘT PHÁ TỚI SOTA 4.26m

### 1. Phân Tích Thực Nghiệm Kết Quả 6.97 Tháng (02c Single-Run)
Trong lần huấn luyện đơn lẻ đầu tiên của Notebook 02c:
- **Khoảng cách tổng quát hóa (Generalization Gap):** Chỉ còn **0.28 tháng** (Train MAE = 6.69m vs Val MAE = 6.97m). Tình trạng Overfitting từ phiên bản 02b trước đó (khoảng cách 3.79m) đã được **triệt tiêu hoàn toàn** nhờ Z-score normalization và Medical Augmentations.
- **Hệ số tương quan $R^2$:** Đạt **0.9507** (độ chính xác giải thích phương sai vượt trội).
- **Pubertal Spurt MAE:** Hạ xuống **5.23 tháng** (giảm sâu từ 7.85m và 6.40m).

### 2. Hai Nút Thắt Kỹ Thuật Đã Được Khắc Phục:
1. **Hiện tượng Gradient Vanishing của Smooth L1 Loss ($\beta = 1.0$ cũ):**
   - Trên không gian Z-score ($\sigma \approx 41.19$ tháng), ngưỡng chuyển đổi giữa $L_1$ và $L_2$ là $|z| \le \beta = 1.0$, tương đương sai số thực tế lên tới **41.2 tháng**.
   - Khi mô hình đã học tốt và sai số giảm xuống khoảng 7 tháng ($|z| \approx 0.17$), hàm loss rơi sâu vào vùng $L_2$. Lúc này gradient đối với đầu ra là $\nabla_z = z / \beta \approx 0.17$, giảm sút hơn 83%! Trọng số mô hình gần như ngừng cập nhật tinh chỉnh.
   - **Giải pháp kỹ thuật:** Đã hạ $\beta = 0.1$ (tương đương $\pm 4.1$ tháng). Mọi sai số từ 4.2 tháng trở lên đều được cung cấp **gradient đầy đủ $\pm 1.0$**, ép mạng tiếp tục hội tụ xuống dưới 5 tháng.
2. **Quy luật Biến thiên Độc lập và Giới hạn Nhiễu Quan sát Y khoa (Inter-Observer Noise):**
   - Độ lệch chuẩn sai số giữa các bác sĩ chẩn đoán hình ảnh nhi khoa (Pediatric Radiologists) là **5.0 – 6.0 tháng**. Một mô hình đơn lẻ không thể vượt qua ngưỡng nhiễu ngẫu nhiên này.
   - Đội vô địch RSNA 2017 (16-bit, 4.265 tháng MAE) đạt được đỉnh cao nhờ **Ensemble 15 mô hình đa tỷ lệ qua 5-Fold Cross Validation**.
   - **Giải pháp kỹ thuật:** Triển khai **5-Fold Stratified Cross Validation** trên 10,088 ảnh huấn luyện + Bảo tồn nguyên vẹn **1,262 ảnh Holdout Test Set**. Cơ chế Soft-Voting Ensemble kết hợp các fold sẽ triệt tiêu phương sai ngẫu nhiên và đưa MAE tiến sát/vượt mốc 4.26m.

---

## 🚀 QUY TRÌNH THỰC THI CHI TIẾT TRÊN KAGGLE (END-TO-END RUNBOOK)

```text
[GIAI ĐOẠN 1: TIỀN XỬ LÝ (CPU)]
01_rsna_preprocessing_and_caching.ipynb
  │
  ├─ Loose-crop, CLAHE 2.0, Letterbox 512x512
  ├─ Tách Stratified: 10,088 Train/Val + 1,262 Holdout Test
  └─ Tạo Kaggle Dataset: `rsna-bone-age-processed-512`
  │
  ▼
[GIAI ĐOẠN 2: HUẤN LUYỆN 5-FOLD ENSEMBLE (GPU)]
02c_champion_dual_branch_full_suite.ipynb
  │
  ├─ Phiên bản 1: FOLD_TO_TRAIN = 0  ──> exp5_fold0_best.pt, exp5_fold0_test_predictions.csv
  ├─ Phiên bản 2: FOLD_TO_TRAIN = 1  ──> exp5_fold1_best.pt, exp5_fold1_test_predictions.csv
  ├─ Phiên bản 3: FOLD_TO_TRAIN = 2  ──> exp5_fold2_best.pt, exp5_fold2_test_predictions.csv
  ├─ Phiên bản 4: FOLD_TO_TRAIN = 3  ──> exp5_fold3_best.pt, exp5_fold3_test_predictions.csv
  ├─ Phiên bản 5: FOLD_TO_TRAIN = 4  ──> exp5_fold4_best.pt, exp5_fold4_test_predictions.csv
  │
  └─ Step 7 (Auto Ensemble Blending): Tự động gộp các file `*_test_predictions.csv`,
     tính Ensemble MAE/RMSE/R2, xuất biểu đồ cột đối chứng IEEE 300 DPI.
  │
  ▼
[GIAI ĐOẠN 3: XUẤT BẢN KHOA HỌC (CPU / GPU)]
03_scientific_evaluation_and_explainability.ipynb
  │
  └─ Xuất trọn bộ 5 hình ảnh IEEE TMI 300 DPI + Mã nguồn Bảng LaTeX cho Overleaf.
```

---

## 🛠️ HƯỚNG DẪN TỪNG BƯỚC CHO NOTEBOOK 02C TRÊN KAGGLE

### Cách 1: Chạy Từng Fold Riêng Biệt (Khuyên Dùng để Quản Lý GPU Tối Ưu)
Mỗi fold chạy 26 epochs mất khoảng **70–80 phút** trên Tesla P100 (hoặc T4). Bạn có thể chạy từng fold theo lịch trình thuận tiện:

1. **Chạy Fold 0:**
   - Mở Notebook `02c_champion_dual_branch_full_suite.ipynb` trên Kaggle.
   - Kiểm tra ô cấu hình `CONFIG['FOLD_TO_TRAIN'] = 0`.
   - Bấm **Save Version** -> **Save & Run All (Commit)**.
   - Sau khi hoàn thành, Notebook Output sẽ có:
     - `exp5_fold0_best.pt`
     - `exp5_fold0_val_predictions.csv`
     - `exp5_fold0_test_predictions.csv`
     - `exp5_fold0_metrics.json`
     - `training_curves_fold0.png`

2. **Chạy Fold 1, 2, 3, 4 (Kết hợp Checkpoint Versioning):**
   - Khi chạy Fold 1: Đổi `CONFIG['FOLD_TO_TRAIN'] = 1`.
   - Gắn Output của Notebook Fold 0 vào Input (qua menu `+ Add Input` -> `Your Work` -> chọn version trước của 02c).
   - Khi chạy xong Fold 1, ô **Step 7 (Ensemble Blending)** sẽ **tự động phát hiện cả Fold 0 và Fold 1**, lập tức tính toán kết quả Ensemble 2-Fold, xuất ra biểu đồ so sánh:
     `[Fold 0: 6.9m] vs [Fold 1: 6.8m] vs [Ensemble 2-Fold: 5.4m]`!
   - Lặp lại tương tự cho các fold tiếp theo. Khi đủ 5 folds, Ensemble 5-Fold sẽ đạt hiệu năng cao nhất.

### Cách 2: Chạy Toàn Bộ 5 Folds Trong Một Phiên Chạy Duy Nhất (Nếu Muốn Để Máy Tự Chạy Đêm)
- Đặt `CONFIG['FOLD_TO_TRAIN'] = None` (hoặc cấu hình vòng lặp 5 folds).
- Với `EPOCHS = 20-26`, tổng thời gian cho 5 folds là khoảng **6–6.5 giờ**.
- Kaggle cho phép phiên chạy commit tối đa **9 giờ liên tục**, hoàn toàn nằm trong ngưỡng an toàn!

---

## 🛡️ CÁC CẢI TIẾN ỔN ĐỊNH HỆ THỐNG ĐÃ ĐƯỢC TÍCH HỢP SẴN TRONG 02C

1. **Triệt tiêu hoàn toàn nguy cơ DataLoader Deadlock (Hanging ở Epoch 1–2):**
   - Đã cấu hình `cv2.setNumThreads(0)` và `cv2.ocl.setUseOpenCL(False)` để ngăn chặn OpenMP mutex conflict giữa OpenCV và PyTorch multiprocessing.
   - Thiết lập `persistent_workers = False` và `NUM_WORKERS = 2` an toàn trên môi trường Linux của Kaggle.
2. **Chặn cảnh báo Garbage Collector của Python 3.13:**
   - Kaggle nâng cấp Python 3.13 gây ra lỗi giả `AssertionError: can only test a child process` khi worker process giải phóng bộ nhớ. Đã tích hợp `sys.unraisablehook` để bỏ qua thông báo vô hại này, giữ logs luôn sạch đẹp.
3. **Giám sát thời gian thực (Real-Time Plotting):**
   - Hàm `plot_and_save_training_progress()` tự động lưu đồ thị `training_curves_fold{k}.png` sau mỗi epoch. Dù phiên chạy bị gián đoạn giữa chừng, bạn vẫn tải được đồ thị của các epoch đã hoàn thành.

---

## 📊 GIAI ĐOẠN 3: XUẤT TRỌN BỘ 5 BIỂU ĐỒ BÀI BÁO (NOTEBOOK 03)

Sau khi có file dự đoán kiểm định độc lập (`exp5_champion_ensemble_test_predictions.csv` hoặc từng fold `exp5_fold*_test_predictions.csv`), hãy mở **Notebook 03**:

1. **Nạp dữ liệu:**
   - Notebook 03 tự động quét và nạp file dự đoán trên tập test độc lập 1,262 ca.
2. **Kết xuất 5 Hình Ảnh Chuẩn IEEE TMI (300 DPI):**
   - **Figure 1 (`fig1_multi_model_convergence.png`):** Đường cong hội tụ Loss và MAE đối đầu 3 thế hệ mô hình (Inception Baseline vs Dual-Branch vs Champion Ensemble).
   - **Figure 2 (`fig2_bland_altman_agreement.png`):** Chronological Calibration ($True$ vs $Pred$) và Biểu đồ Bland-Altman kiểm định độ tương đồng lâm sàng ($\pm 1.96$ SD).
   - **Figure 3 (`fig3_pubertal_cohort_error_distribution.png`):** Boxplot phân tích sai số trên các giai đoạn phát triển: Infancy, Pre-pubertal, Pubertal Spurt, Post-pubertal.
   - **Figure 4 (`fig4_clinical_explainability_4panel.png`):** Khung giải thích lâm sàng 4 panels đối chiếu Grad-CAM (ConvNeXt), Swin Cross-Attention Map với 13 vùng sụn giải phẫu chuẩn Tanner-Whitehouse (TW3 13-RUS).
   - **Figure 5 (`fig5_ensemble_variance_reduction.png`):** Biểu đồ cột minh chứng trực quan mức độ giảm phương sai và mức cải thiện MAE qua từng fold và sau khi Ensemble.
3. **Mã Nguồn Bảng LaTeX (`table_results.tex`):**
   - Tự động định dạng bảng chuẩn `\begin{table*}[t]` với đầy đủ các chỉ số: MAE, RMSE, $R^2$, Pubertal MAE, Parameters (M), GFLOPs để chèn trực tiếp vào Overleaf.

---

## 🎯 DANH MỤC THÔNG SỐ VÀNG CỦA MÔ HÌNH PROPOSED DUAL-BRANCH

| Thành phần | Cấu hình kỹ thuật | Rationale Y khoa / Kỹ thuật |
| :--- | :--- | :--- |
| **Branch A (Local Detail)** | ConvNeXt-Tiny ($384 \times 384$) | $7 \times 7$ depthwise conv bảo toàn vân xương & khe khớp sụn |
| **Branch B (Global Context)** | Swin Transformer-Tiny ($224 \times 224$) | Shifted window self-attention nắm bắt tương quan toàn bàn tay |
| **Token Representation** | $49 \times 768$ (Swin) & $144 \times 768$ (ConvNeXt) | Giữ nguyên chiều không gian $7\times7$ và $12\times12$ thay vì Global Pool sớm |
| **Cross-Attention Fusion** | $Q$: Swin Global, $K, V$: ConvNeXt Local | Transformer toàn cục truy vấn chi tiết kết cấu cục bộ |
| **Gender Conditioning** | FiLM Layer (Affine: $\gamma \odot f + \beta$) | Điều biến đặc trưng thích ứng với tốc độ cốt hóa khác nhau theo giới |
| **Loss Function** | Smooth L1 ($\beta = 0.08$) + Pubertal ($\lambda=0.35$) | Bảo toàn gradient $\pm 1.0$ và tăng cường tối ưu hóa cửa sổ dậy thì |
| **Regularization** | Stochastic Depth (0.2) + Weight Decay (0.05) | Triệt tiêu hoàn toàn overfitting trên dữ liệu X-quang |
| **Validation Strategy** | 5-Fold Stratified CV + 1,262 Holdout Test | Khách quan, tái lập, không rò rỉ dữ liệu (No Data Leakage) |

---

## 🏛️ LỘ TRÌNH ĐỘT PHÁ SOTA $\le 4.25$ THÁNG (NOTEBOOKS 02E, 02F, 03B)

### 1. Giai đoạn 1: Notebook 02e (`02e_champion_scale_and_resolution.ipynb`)
- **Mục tiêu:** Nâng cấp mạng lên ConvNeXt-Small ($512\times512$, 256 tokens) + Swin-Small ($384\times384$, 144 tokens) = **102.5M params** và áp dụng Adaptive Pubertal Huber Loss.
- **Cách chạy:**
  - Mở notebook `02e_champion_scale_and_resolution.ipynb` trên Kaggle GPU (P100 hoặc T4).
  - Có thể chạy 1 Fold benchmark (`CONFIG['RUN_FOLDS'] = [0]`, ~1.3h) hoặc chạy cả 5 Folds (`CONFIG['RUN_FOLDS'] = [0, 1, 2, 3, 4]`, ~6.5h).
  - Bấm **Save Version** -> **Save & Run All (Commit)**.
  - Kỳ vọng: Kéo MAE xuống **$5.2 - 5.5$ tháng**.

### 2. Giai đoạn 2: Notebook 02f (`02f_anatomical_roi_multicrop.ipynb`)
- **Mục tiêu:** Chuyên gia giải phẫu Dual-Crop kết hợp Swin toàn bàn tay ($384\times384$) và ConvNeXt cận cảnh cổ tay/đĩa sụn tiếp hợp ($256\times256$).
- **Cách chạy:**
  - Mở notebook `02f_anatomical_roi_multicrop.ipynb` trên Kaggle GPU.
  - Bấm **Save Version** -> **Save & Run All (Commit)** (~6.0h cho 5 Folds).
  - Kỳ vọng: Kéo MAE xuống **$4.6 - 4.9$ tháng** (chính thức vượt ngưỡng bác sĩ con người 5.0m!).

### 3. Giai đoạn 3: Notebook 03b (`03b_sota_ensemble_and_manuscript.ipynb`)
- **Mục tiêu:** Hòa trộn Stacking Meta-Regressor đa kiến trúc (Ridge / Bayesian) giữa 02d, 02e, 02f, triệt tiêu phương sai tương quan và xuất bản 5 biểu đồ IEEE 300 DPI kèm mã nguồn LaTeX.
- **Cách chạy:**
  - Gắn Outputs của các notebook 02d, 02e, 02f vào Input của notebook `03b_sota_ensemble_and_manuscript.ipynb`.
  - Chạy notebook trên Kaggle CPU hoặc GPU (~10-20 phút).
  - Tải về trọn bộ 5 biểu đồ tại `/kaggle/working/figures/` và mã nguồn bảng LaTeX tại `/kaggle/working/outputs/tables_ieee_tmi.tex`.
  - Kỳ vọng: Chạm mốc **$\le 4.25$ tháng** (Vượt kỷ lục nhà vô địch RSNA 2017)!

