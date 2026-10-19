"""Script to generate Notebook 02a: 02a_inception_rsna2017_baseline.ipynb."""

import json
import os

def create_notebook():
    cells = []

    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Header
    add_md("""# 🏆 MỐC 1: TÁI LẬP NHÀ VÔ ĐỊCH RSNA 2017 BASELINE (INCEPTION-V3 + GENDER CONCAT)
### Đề tài: *Harmonizing Local Inductive Bias and Global Contextual Reasoning: A Dual-Branch ConvNeXt-Swin Transformer Architecture with Gender-Modulated Cross-Attention for Pediatric Bone Age Assessment*

---
## 🎯 Mục Tiêu Khoa Học & Ý Nghĩa Câu Chuyện (Milestone 1):
- **Tái lập trung thực mốc chuẩn đối chứng (Fair Benchmarking):** Huấn luyện kiến trúc đoạt giải nhất cuộc thi RSNA 2017 (Inception-v3 kết hợp ghép nối thô vector giới tính) trên cùng tập dữ liệu phân tầng 80-10-10 đã tiền xử lý ở Notebook 01.
- **Xác lập mốc sai số nội bộ (Internal Baseline MAE):** Mọi tuyên bố cải tiến ở các mô hình nhánh kép sau này chỉ có giá trị khi so sánh trên cùng một phân phối dữ liệu với mốc Inception-v3 này.
- **Lưu trữ dẫn chứng khoa học thật:** Tự động xuất biểu đồ hội tụ, file lịch sử `exp0_history.csv`, file dự đoán `exp0_val_predictions.csv`, và checkpoint trọng số thật `exp0_inception_best.pt`.""")

    # Pip install timm
    add_code("""# Cài đặt thư viện phụ trợ (nếu môi trường Kaggle chưa có)
!pip install -q timm albumentations""")

    # Imports & Config
    add_code("""# ==============================================================================
# BƯỚC 1: CẤU HÌNH SIÊU THAM SỐ TẬP TRUNG & MÔI TRƯỜNG THỰC THI
# ==============================================================================
import os
import sys
import time
import math
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
import timm

# Thiết lập hiển thị đồ họa
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# BẢNG CẤU HÌNH THÍ NGHIỆM EXP 0 (INCEPTION-V3 BASELINE)
CONFIG = {
    'EXPERIMENT': 'exp0_inception_baseline',
    'SEED': 42,
    'IMAGE_SIZE': 299,          # Kích thước chuẩn của Inception-v3
    'BATCH_SIZE': 16,
    'NUM_WORKERS': 4,
    'EPOCHS': 15,
    'LR': 2e-4,
    'WEIGHT_DECAY': 1e-4,
    'GRADIENT_ACCUMULATION': 2, # Effective Batch Size = 32
    'USE_AMP': True,
    'CHECKPOINT_DIR': '/kaggle/working/checkpoints',
    'OUTPUT_DIR': '/kaggle/working/outputs',
}

# Cố định Seed đảm bảo tính tái lập 100%
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG['SEED'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Môi trường phần cứng: {device} | PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"🚀 Tên GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

os.makedirs(CONFIG['CHECKPOINT_DIR'], exist_ok=True)
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)""")

    # Auto find preprocessed dataset
    add_md("""---
## 📂 BƯỚC 2: TỰ ĐỘNG NHẬN DIỆN DATASET ĐÃ TIỀN XỬ LÝ (TỪ NOTEBOOK 01)
Tự động tìm kiếm thư mục chứa ảnh $512\\times512$ và file `train_stratified.csv` được gắn vào tab Input của Kaggle.""")

    add_code("""# ==============================================================================
# BƯỚC 2: QUÉT ĐƯỜNG DẪN DỮ LIỆU ĐÃ TIỀN XỬ LÝ
# ==============================================================================
DATASET_DIR = None
STRATIFIED_CSV_PATH = None
IMAGES_DIR = None

# Tìm kiếm trong /kaggle/input
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f == 'train_stratified.csv':
            STRATIFIED_CSV_PATH = Path(root) / f
            DATASET_DIR = Path(root)
            break

if STRATIFIED_CSV_PATH is None:
    # Fallback kiểm tra /kaggle/working hoặc local
    if Path('/kaggle/working/rsna_processed_512/train_stratified.csv').exists():
        STRATIFIED_CSV_PATH = Path('/kaggle/working/rsna_processed_512/train_stratified.csv')
        DATASET_DIR = Path('/kaggle/working/rsna_processed_512')
    elif Path('./rsna_processed_512/train_stratified.csv').exists():
        STRATIFIED_CSV_PATH = Path('./rsna_processed_512/train_stratified.csv')
        DATASET_DIR = Path('./rsna_processed_512')

if STRATIFIED_CSV_PATH is None:
    raise FileNotFoundError(
        "❌ CHƯA TÌM THẤY TỆP 'train_stratified.csv'!\\n"
        "👉 Cách khắc phục: Ở cột bên phải (Input), nhấn '+ Add Input' -> tìm Dataset bạn đã lưu từ Notebook 01 -> nhấn Add."
    )

IMAGES_DIR = DATASET_DIR / 'images'
print(f"✅ Đã kết nối Dataset thành công:")
print(f"• File phân tầng: {STRATIFIED_CSV_PATH}")
print(f"• Thư mục ảnh:    {IMAGES_DIR}")

df_all = pd.read_csv(STRATIFIED_CSV_PATH)
train_df = df_all[df_all['split'] == 'train'].reset_index(drop=True)
val_df = df_all[df_all['split'] == 'val'].reset_index(drop=True)
test_df = df_all[df_all['split'] == 'test'].reset_index(drop=True)

print(f"📊 Phân bổ mẫu huấn luyện:")
print(f"• Train set: {len(train_df):,} ảnh")
print(f"• Val set:   {len(val_df):,} ảnh")
print(f"• Test set:  {len(test_df):,} ảnh")""")

    # Dataset & Loader
    add_md("""---
## 📦 BƯỚC 3: XÂY DỰNG DATASET & TRANSFORMS PHÙ HỢP INCEPTION-V3
Inception-v3 yêu cầu đầu vào ảnh kích thước chuẩn $299\\times299$ với 3 kênh màu.""")

    add_code("""# ==============================================================================
# BƯỚC 3: XÂY DỰNG PYTORCH DATASET CHO INCEPTION-V3
# ==============================================================================
class RSNABaselineDataset(Dataset):
    def __init__(self, df, img_dir, img_size=299, is_train=True):
        self.df = df
        self.img_dir = img_dir
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = int(row['id'])
        img_path = self.img_dir / f"{img_id}.png"
        
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        # Chuyển sang 3 kênh màu chuẩn ImageNet
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Chuẩn hóa về [0, 1] và trừ chuẩn ImageNet
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        gender = torch.tensor([1.0 if row['male'] else 0.0], dtype=torch.float32)
        boneage = torch.tensor(row['boneage'], dtype=torch.float32)

        return img_tensor, gender, boneage

train_dataset = RSNABaselineDataset(train_df, IMAGES_DIR, img_size=CONFIG['IMAGE_SIZE'], is_train=True)
val_dataset = RSNABaselineDataset(val_df, IMAGES_DIR, img_size=CONFIG['IMAGE_SIZE'], is_train=False)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, 
                          num_workers=CONFIG['NUM_WORKERS'], pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, 
                        num_workers=CONFIG['NUM_WORKERS'], pin_memory=True)

print(f"✅ Khởi tạo DataLoader thành công: {len(train_loader)} batches/epoch")""")

    # Model Architecture
    add_md("""---
## 🧠 BƯỚC 4: KIẾN TRÚC MÔ HÌNH INCEPTION-V3 BASELINE (RSNA 2017 CHAMPION)
Mô phỏng chính xác kiến trúc đoạt giải RSNA 2017:
- Trích xuất đặc trưng bằng mạng tích chập Inception-v3 đa tỷ lệ ($1\\times1, 3\\times3, 5\\times5$).
- Lớp Global Average Pooling cho vector 2048 chiều.
- Ghép nối vector giới tính: $\\mathbf{f}_{\\text{fused}} = [\\mathbf{f}_{\\text{visual}} \\parallel \\mathbf{g}] \\in \\mathbb{R}^{2049}$.
- Regression Head dự đoán trực tiếp số tháng tuổi.""")

    add_code("""# ==============================================================================
# BƯỚC 4: ĐỊNH NGHĨA MÔ HÌNH INCEPTION-V3 VỚI NAIVE GENDER CONCAT
# ==============================================================================
class InceptionV3BoneAgeBaseline(nn.Module):
    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        # Sử dụng timm: tự động tải trọng số pre-trained, vô hiệu hóa aux_logits và trả về feature vector 2048 chiều chuẩn
        self.inception = timm.create_model('inception_v3', pretrained=pretrained, num_classes=0)
        in_features = getattr(self.inception, 'num_features', 2048)

        # Đầu hồi quy với Naive Concatenation: 2048 + 1 (giới tính) = 2049
        self.regressor = nn.Sequential(
            nn.Linear(in_features + 1, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, gender):
        # Trích xuất đặc trưng từ Inception-v3
        feat = self.inception(x)  # [B, 2048]
        # Ghép nối thô vector giới tính
        fused = torch.cat([feat, gender], dim=-1)  # [B, 2049]
        out = self.regressor(fused)
        return out.squeeze(-1)

model = InceptionV3BoneAgeBaseline(pretrained=True).to(device)
total_params = sum(p.parameters().count() if hasattr(p, 'count') else p.numel() for p in model.parameters())
print(f"✅ Khởi tạo InceptionV3BoneAgeBaseline thành công! Tổng tham số: {total_params/1e6:.2f}M")""")

    # Training Loop with Real-Time Plots
    add_md("""---
## 🔄 BƯỚC 5: TRAINING ENGINE VỚI VẼ ĐỒ THỊ TRỰC TIẾP & LƯU LOG THẬT
- Smooth L1 Loss (Huber Loss với $\\beta=1.0$) ổn định gradient.
- Automatic Mixed Precision (AMP FP16) tăng tốc độ x2 trên GPU.
- Gradient Accumulation ($k=2$).
- Vẽ biểu đồ Loss và MAE trực tiếp sau mỗi epoch trong output.
- Lưu lại toàn bộ lịch sử ra `outputs/exp0_history.csv` và checkpoint ra `checkpoints/exp0_inception_best.pt`.""")

    add_code("""# ==============================================================================
# BƯỚC 5: HUẤN LUYỆN MÔ HÌNH VỚI THEO DÕI THỜI GIAN THỰC
# ==============================================================================
criterion = nn.SmoothL1Loss(beta=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LR'], weight_decay=CONFIG['WEIGHT_DECAY'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS'], eta_min=1e-6)
scaler = GradScaler(enabled=CONFIG['USE_AMP'])

history = {
    'epoch': [], 'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': [], 'lr': []
}

best_val_mae = float('inf')
checkpoint_path = Path(CONFIG['CHECKPOINT_DIR']) / f"{CONFIG['EXPERIMENT']}_best.pt"

print(f"🚀 BẮT ĐẦU HUẤN LUYỆN {CONFIG['EPOCHS']} EPOCHS...")

for epoch in range(1, CONFIG['EPOCHS'] + 1):
    start_time = time.time()
    model.train()
    total_train_loss, total_train_mae, train_samples = 0.0, 0.0, 0
    optimizer.zero_grad()

    # Vòng lặp huấn luyện
    for step, (images, genders, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch:02d}/{CONFIG['EPOCHS']:02d} [Train]")):
        images, genders, targets = images.to(device, non_blocking=True), genders.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        with autocast(enabled=CONFIG['USE_AMP']):
            preds = model(images, genders)
            loss = criterion(preds, targets) / CONFIG['GRADIENT_ACCUMULATION']

        scaler.scale(loss).backward()

        if (step + 1) % CONFIG['GRADIENT_ACCUMULATION'] == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        batch_sz = targets.size(0)
        total_train_loss += loss.item() * CONFIG['GRADIENT_ACCUMULATION'] * batch_sz
        total_train_mae += torch.sum(torch.abs(preds.detach() - targets)).item()
        train_samples += batch_sz

    scheduler.step()
    epoch_train_loss = total_train_loss / train_samples
    epoch_train_mae = total_train_mae / train_samples

    # Vòng lặp kiểm định
    model.eval()
    total_val_loss, total_val_mae, val_samples = 0.0, 0.0, 0

    with torch.no_grad():
        for images, genders, targets in val_loader:
            images, genders, targets = images.to(device), genders.to(device), targets.to(device)
            with autocast(enabled=CONFIG['USE_AMP']):
                preds = model(images, genders)
                loss = criterion(preds, targets)

            batch_sz = targets.size(0)
            total_val_loss += loss.item() * batch_sz
            total_val_mae += torch.sum(torch.abs(preds - targets)).item()
            val_samples += batch_sz

    epoch_val_loss = total_val_loss / val_samples
    epoch_val_mae = total_val_mae / val_samples
    elapsed = time.time() - start_time
    curr_lr = optimizer.param_groups[0]['lr']

    # Ghi lại lịch sử
    history['epoch'].append(epoch)
    history['train_loss'].append(epoch_train_loss)
    history['train_mae'].append(epoch_train_mae)
    history['val_loss'].append(epoch_val_loss)
    history['val_mae'].append(epoch_val_mae)
    history['lr'].append(curr_lr)

    # Lưu checkpoint tốt nhất
    is_best = epoch_val_mae < best_val_mae
    if is_best:
        best_val_mae = epoch_val_mae
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_val_mae': best_val_mae,
            'config': CONFIG
        }, checkpoint_path)

    print(f"Epoch [{epoch:02d}/{CONFIG['EPOCHS']:02d}] "
          f"Train Loss: {epoch_train_loss:.4f} | Train MAE: {epoch_train_mae:.2f}m | "
          f"Val Loss: {epoch_val_loss:.4f} | Val MAE: {epoch_val_mae:.2f}m | "
          f"Best Val MAE: {best_val_mae:.2f}m {'🌟' if is_best else ''} ({elapsed:.1f}s)")

    # Vẽ đồ thị thu nhỏ tiến trình trực tiếp
    if epoch % 1 == 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
        axes[0].plot(history['epoch'], history['train_loss'], label='Train Loss', color='blue')
        axes[0].plot(history['epoch'], history['val_loss'], label='Val Loss', color='red', linestyle='--')
        axes[0].set_title("Loss (Smooth L1)")
        axes[0].legend()
        
        axes[1].plot(history['epoch'], history['train_mae'], label='Train MAE', color='blue')
        axes[1].plot(history['epoch'], history['val_mae'], label='Val MAE', color='red', linestyle='--')
        axes[1].axhline(best_val_mae, color='green', linestyle=':', label=f"Best: {best_val_mae:.2f}m")
        axes[1].set_title("Mean Absolute Error (Months)")
        axes[1].legend()
        plt.tight_layout()
        plt.show()""")

    # Evaluation on Validation/Test Set
    add_md("""---
## 📊 BƯỚC 6: XUẤT TOÀN BỘ KẾT QUẢ THẬT (LOGS, PREDICTIONS & METRICS)
Tải trọng số tốt nhất `exp0_inception_best.pt` để tính toán các chỉ số thống kê lâm sàng (MAE, RMSE, $R^2$, 95% Bootstrap CI) và lưu lại các file `.csv` phục vụ Notebook 03.""")

    add_code("""# ==============================================================================
# BƯỚC 6: ĐÁNH GIÁ CHUYÊN SÂU VỚI CHECKPOINT TỐT NHẤT
# ==============================================================================
# Nạp lại checkpoint tốt nhất
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ Đã nạp thành công checkpoint tốt nhất từ Epoch {checkpoint['epoch']} với Val MAE: {checkpoint['best_val_mae']:.2f} tháng")

# Thu thập dự đoán chi tiết trên tập Validation
val_results = []
with torch.no_grad():
    for idx in range(len(val_df)):
        row = val_df.iloc[idx]
        img_tensor, gender, boneage = val_dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        gender = gender.unsqueeze(0).to(device)

        with autocast(enabled=CONFIG['USE_AMP']):
            pred = model(img_tensor, gender).item()

        val_results.append({
            'id': int(row['id']),
            'male': bool(row['male']),
            'boneage_true': float(boneage.item()),
            'boneage_pred': float(pred),
            'abs_error': abs(float(pred) - float(boneage.item()))
        })

df_preds = pd.DataFrame(val_results)

# Tính toán các chỉ số thống kê chuẩn
errors = df_preds['abs_error'].values
overall_mae = np.mean(errors)
overall_rmse = np.sqrt(np.mean((df_preds['boneage_true'] - df_preds['boneage_pred'])**2))
ss_res = np.sum((df_preds['boneage_true'] - df_preds['boneage_pred'])**2)
ss_tot = np.sum((df_preds['boneage_true'] - np.mean(df_preds['boneage_true']))**2)
r2_score = 1.0 - (ss_res / ss_tot)

# Sai số nhóm tuổi dậy thì (Nữ: 11-14y, Nam: 12-15y)
is_puberty_female = (~df_preds['male']) & (df_preds['boneage_true'] >= 132) & (df_preds['boneage_true'] <= 168)
is_puberty_male = (df_preds['male']) & (df_preds['boneage_true'] >= 144) & (df_preds['boneage_true'] <= 180)
puberty_df = df_preds[is_puberty_female | is_puberty_male]
puberty_mae = puberty_df['abs_error'].mean() if len(puberty_df) > 0 else 0.0

# Bootstrap 1000 lần tính khoảng tin cậy 95%
bootstrap_maes = []
for _ in range(1000):
    sample = np.random.choice(errors, size=len(errors), replace=True)
    bootstrap_maes.append(np.mean(sample))
ci_lower, ci_upper = np.percentile(bootstrap_maes, [2.5, 97.5])

metrics_summary = {
    'experiment': CONFIG['EXPERIMENT'],
    'best_epoch': checkpoint['epoch'],
    'overall_mae': float(overall_mae),
    'mae_ci_95': [float(ci_lower), float(ci_upper)],
    'rmse': float(overall_rmse),
    'r2_score': float(r2_score),
    'male_mae': float(df_preds[df_preds['male']]['abs_error'].mean()),
    'female_mae': float(df_preds[~df_preds['male']]['abs_error'].mean()),
    'puberty_mae': float(puberty_mae),
}

# Lưu các tệp kết quả ra thư mục outputs/
history_csv_path = Path(CONFIG['OUTPUT_DIR']) / f"{CONFIG['EXPERIMENT']}_history.csv"
preds_csv_path = Path(CONFIG['OUTPUT_DIR']) / f"{CONFIG['EXPERIMENT']}_val_predictions.csv"
metrics_json_path = Path(CONFIG['OUTPUT_DIR']) / f"{CONFIG['EXPERIMENT']}_metrics.json"

pd.DataFrame(history).to_csv(history_csv_path, index=False)
df_preds.to_csv(preds_csv_path, index=False)
with open(metrics_json_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_summary, f, indent=2)

print("=" * 60)
print(f"📋 TỔNG KẾT KẾT QUẢ EXP 0 (INCEPTION-V3 BASELINE):")
print(f"• Overall MAE:       {overall_mae:.2f} tháng (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])")
print(f"• Overall RMSE:      {overall_rmse:.2f} tháng")
print(f"• R2 Score:          {r2_score:.4f}")
print(f"• Pubertal Spurt MAE:{puberty_mae:.2f} tháng (N={len(puberty_df)})")
print(f"💾 Đã lưu file lịch sử:    {history_csv_path}")
print(f"💾 Đã lưu file dự đoán:    {preds_csv_path}")
print(f"💾 Đã lưu file chỉ số:     {metrics_json_path}")
print(f"💾 Đã lưu file checkpoint: {checkpoint_path}")
print("=" * 60)""")

    nb_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_file = "d:/Development/projects/dual_branch_bone_age/kaggle_notebooks/02a_inception_rsna2017_baseline.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1, ensure_ascii=False)
    print(f"Generated {out_file} successfully!")

if __name__ == "__main__":
    create_notebook()
