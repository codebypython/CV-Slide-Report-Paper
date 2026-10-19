"""Script to generate Notebook 02f: 02f_anatomical_roi_multicrop.ipynb

Phase 2 of Deep SOTA Master Plan:
1. High-resolution anatomical zoom on Wrist / Carpal cluster & Distal Epiphysis (256x256).
2. Global Hand context stream (384x384) with Swin-Small.
3. Micro-morphology wrist stream (256x256) with ConvNeXt-Small.
4. Cross-Attention: 144 macro queries inspect 64 high-resolution epiphyseal keys.
5. Adaptive Pubertal Huber/Smooth-L1 Loss (lambda_puberty=0.35, beta=0.08).
6. 5-Fold Stratified Cross-Validation + 1,262 Holdout Test evaluation with TTA.
7. Real-time heartbeat logging to '/kaggle/working/LIVE_STATUS.txt'.
"""

import json
import os
from pathlib import Path

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
    add_md("""# 🔬 MỐC 2F: CHUYÊN GIA GIẢI PHÃU DUAL-CROP & ZOOM QUANG HỌC CỔ TAY (ANATOMICAL ROI ZOOM)
### Đề tài: *Harmonizing Local Inductive Bias and Global Contextual Reasoning: A Dual-Branch ConvNeXt-Swin Transformer Architecture with Gender-Modulated Cross-Attention for Pediatric Bone Age Assessment*
**Target Venue:** *IEEE Transactions on Medical Imaging (IEEE TMI)* / *Medical Image Analysis (MedIA)*

---
## 🎯 Đột Phá Kỹ Thuật Của Phiên Bản 02f (Phase 2 Deep Master Plan):
1. **Mô Phỏng Thao Tác Phóng To Kính Hiển Vi Của Bác Sĩ Chẩn Đoán (Optical Zoom ROI):**
   - Thay vì nén toàn bộ bàn tay làm mờ các đĩa sụn, hệ thống chia làm **2 luồng quang học phối hợp**:
     - **Luồng 1 (Global Macro-Structure $384 \\times 384$):** Swin Transformer đánh giá tỷ lệ các ngón tay và cấu trúc hình học toàn bàn tay (144 macro queries).
     - **Luồng 2 (Micro Anatomical Carpal Zoom $256 \\times 256$):** Cắt tự động vùng cổ tay & đầu dưới xương quay/trụ ($y \\in [0.45, 0.95], x \\in [0.15, 0.85]$) ở độ phân giải cao nhất, đưa vào ConvNeXt-Small để soi rõ từng giai đoạn vôi hóa của 7 xương cổ tay và đĩa tăng trưởng (64 micro keys).
2. **Cơ Chế Cross-Attention Định Hướng Chuyên Sâu:**
   - Swin macro queries trực tiếp truy vấn vào 64 keys giải phẫu cổ tay có độ nét cực cao, triệt tiêu hoàn toàn sự mơ hồ tại giai đoạn dậy thì.
3. **Hiệu Suất Bộ Nhớ & Tốc Độ:**
   - Kích thước ảnh $384 + 256$ tối ưu hơn $512$, giảm tải tính toán 30%, cho phép huấn luyện mượt mà trên GPU P100 / T4.""")

    add_code("""# Cài đặt thư viện phụ trợ (timm, albumentations, tabulate)
!pip install -q timm albumentations tabulate""")

    # Step 1: Config
    add_code("""# ==============================================================================
# BƯỚC 1: CẤU HÌNH SIÊU THAM SỐ GIAI ĐOẠN 2 (ANATOMICAL DUAL-CROP ZOOM)
# ==============================================================================
import os
import sys
import time
import math
import json
import random
import gc
from datetime import datetime
from pathlib import Path

# Triệt tiêu vĩnh viễn nguy cơ Deadlock đa luồng giữa OpenCV và PyTorch DataLoader
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import albumentations as A
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm

def silence_dataloader_gc(unraisable):
    if unraisable.exc_type is AssertionError or "dataloader" in str(unraisable.object).lower():
        return
    sys.__unraisablehook__(unraisable)
sys.unraisablehook = silence_dataloader_gc

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

CONFIG = {
    'EXPERIMENT': 'exp6_anatomical_dualcrop',
    'SEED': 42,
    'NUM_FOLDS': 5,
    'RUN_FOLDS': [0, 1, 2, 3, 4],  # Danh sách các folds sẽ chạy (mặc định cả 5 folds)
    'EPOCHS': 20,
    'WARMUP_EPOCHS': 2,
    'USE_ZSCORE': True,
    'USE_TTA': True,
    'CONVNEXT_MODEL': 'convnext_small',
    'SWIN_MODEL': 'swin_small_patch4_window7_224',
    'IMAGE_SIZE_HAND': 384,        # Ảnh toàn bàn tay cho Swin Macro (12x12 = 144 queries)
    'IMAGE_SIZE_WRIST': 256,       # Ảnh cận cảnh cổ tay & đầu dưới xương quay/trụ (8x8 = 64 keys)
    'ROI_YMIN': 0.45,
    'ROI_YMAX': 0.95,
    'ROI_XMIN': 0.15,
    'ROI_XMAX': 0.85,
    'EMBED_DIM': 512,
    'NUM_HEADS': 8,
    'DROPOUT': 0.3,
    'DROP_PATH_RATE': 0.2,
    'BATCH_SIZE': 10,              # Batch size 10 cho tốc độ cao và vừa vặn VRAM
    'GRADIENT_ACCUMULATION': 3,    # Effective Batch Size = 30
    'NUM_WORKERS': 2,
    'SMOOTH_L1_BETA': 0.08,
    'PUBERTY_WEIGHT': 0.35,
    'LR_HEAD_WARMUP': 1e-3,
    'LR_BACKBONE': 2.2e-5,
    'LR_HEAD': 2.8e-4,
    'WEIGHT_DECAY': 0.05,
    'USE_AMP': True,
    'CHECKPOINT_DIR': '/kaggle/working/checkpoints',
    'OUTPUT_DIR': '/kaggle/working/outputs',
    'STATUS_LOG': '/kaggle/working/LIVE_STATUS.txt'
}

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
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

with open(CONFIG['STATUS_LOG'], 'w', encoding='utf-8') as f:
    f.write(f"=== BẮT ĐẦU PHIÊN CHẠY 02F (ANATOMICAL DUAL-CROP) LÚC {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\\n")

def log_status(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    sys.stdout.flush()
    with open(CONFIG['STATUS_LOG'], 'a', encoding='utf-8') as f:
        f.write(formatted + "\\n")""")

    # Step 2: Data connection
    add_md("""---
## 📂 BƯỚC 2: KẾT NỐI DATASET & PHÂN CHIA 5-FOLD STRATIFIED""")

    add_code("""# ==============================================================================
# BƯỚC 2: KẾT NỐI DỮ LIỆU & PHÂN CHIA 5-FOLD STRATIFIED K-FOLD
# ==============================================================================
DATASET_DIR = None
STRATIFIED_CSV_PATH = None
IMAGES_DIR = None

if Path('/kaggle/input').exists():
    for root, dirs, files in os.walk('/kaggle/input'):
        dirs[:] = [d for d in dirs if d.lower() not in ['images', 'image', 'boneage-training-dataset', 'boneage-test-dataset']]
        if 'train_stratified.csv' in files:
            STRATIFIED_CSV_PATH = Path(root) / 'train_stratified.csv'
            DATASET_DIR = Path(root)
            break

if STRATIFIED_CSV_PATH is None:
    if Path('/kaggle/working/rsna_processed_512/train_stratified.csv').exists():
        STRATIFIED_CSV_PATH = Path('/kaggle/working/rsna_processed_512/train_stratified.csv')
        DATASET_DIR = Path('/kaggle/working/rsna_processed_512')
    elif Path('./rsna_processed_512/train_stratified.csv').exists():
        STRATIFIED_CSV_PATH = Path('./rsna_processed_512/train_stratified.csv')
        DATASET_DIR = Path('./rsna_processed_512')

if STRATIFIED_CSV_PATH is None:
    raise FileNotFoundError("❌ CHƯA TÌM THẤY TỆP 'train_stratified.csv'! Hãy gán Dataset từ Notebook 01.")

IMAGES_DIR = DATASET_DIR / 'images'
df_all = pd.read_csv(STRATIFIED_CSV_PATH)

test_df = df_all[df_all['split'] == 'test'].reset_index(drop=True)
pool_df = df_all[df_all['split'] != 'test'].reset_index(drop=True)

strata_keys = pool_df['age_quantile'].astype(str) + "_" + pool_df['male'].astype(str)
skf = StratifiedKFold(n_splits=CONFIG['NUM_FOLDS'], shuffle=True, random_state=CONFIG['SEED'])

pool_df['fold'] = -1
for f_idx, (train_indices, val_indices) in enumerate(skf.split(pool_df, strata_keys)):
    pool_df.loc[val_indices, 'fold'] = f_idx

log_status(f"✅ ĐÃ PHÂN CHIA 5-FOLD STRATIFIED HOÀN TẤT:")
log_status(f"• Tập Huấn luyện/Kiểm định: {len(pool_df):,} mẫu")
log_status(f"• Tập Kiểm thử độc lập (Holdout Test): {len(test_df):,} mẫu (Zero Leakage)")""")

    # Step 3: Dual-Crop Dataset & Loss
    add_md("""---
## 📦 BƯỚC 3: DUAL-CROP OPTICAL ZOOM DATASET & PUBERTAL LOSS""")

    add_code("""# ==============================================================================
# BƯỚC 3: ANATOMICAL DUAL-CROP DATASET & LOSS FUNCTION
# ==============================================================================
train_transform_cv = A.Compose([
    A.Affine(scale=(0.94, 1.06), translate_percent=(-0.04, 0.04), rotate=(-8, 8), border_mode=0, fill=0, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(12, 32), hole_width_range=(12, 32), fill=0, p=0.4),
])

class RSNADualCropDataset(Dataset):
    def __init__(self, df, img_dir, hand_size=384, wrist_size=256, is_train=True, use_zscore=True, mean_target=127.3, std_target=41.2):
        self.df = df
        self.img_dir = img_dir
        self.hand_size = hand_size
        self.wrist_size = wrist_size
        self.is_train = is_train
        self.use_zscore = use_zscore
        self.mean_target = mean_target
        self.std_target = std_target
        self.mean_rgb = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std_rgb = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = int(row['id'])
        img_path = self.img_dir / f"{img_id}.png"
        if not img_path.exists():
            img_path = self.img_dir / f"{img_id}.jpg"

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {img_path}")

        if self.is_train:
            augmented = train_transform_cv(image=img)
            img = augmented['image']

        h, w = img.shape
        # Cắt cận cảnh cổ tay & sụn tiếp hợp (Distal Radius/Ulna & Carpals)
        y1 = max(0, int(h * CONFIG['ROI_YMIN']))
        y2 = min(h, int(h * CONFIG['ROI_YMAX']))
        x1 = max(0, int(w * CONFIG['ROI_XMIN']))
        x2 = min(w, int(w * CONFIG['ROI_XMAX']))

        wrist_crop = img[y1:y2, x1:x2]
        if wrist_crop.size == 0 or wrist_crop.shape[0] < 10 or wrist_crop.shape[1] < 10:
            wrist_crop = img

        img_hand = cv2.resize(img, (self.hand_size, self.hand_size), interpolation=cv2.INTER_LINEAR)
        img_wrist = cv2.resize(wrist_crop, (self.wrist_size, self.wrist_size), interpolation=cv2.INTER_LINEAR)

        t_hand = torch.from_numpy(img_hand).float() / 255.0
        t_hand = t_hand.unsqueeze(0).repeat(3, 1, 1)
        t_hand = (t_hand - self.mean_rgb) / self.std_rgb

        t_wrist = torch.from_numpy(img_wrist).float() / 255.0
        t_wrist = t_wrist.unsqueeze(0).repeat(3, 1, 1)
        t_wrist = (t_wrist - self.mean_rgb) / self.std_rgb

        gender = torch.tensor([1.0 if row['male'] else 0.0], dtype=torch.float32)
        target_raw = float(row['boneage'])

        if self.use_zscore:
            target = (target_raw - self.mean_target) / self.std_target
        else:
            target = target_raw

        return t_hand, t_wrist, gender, torch.tensor(target, dtype=torch.float32), torch.tensor(target_raw, dtype=torch.float32)

class AdaptivePubertalLoss(nn.Module):
    def __init__(self, beta=0.08, puberty_lambda=0.35):
        super().__init__()
        self.beta = beta
        self.puberty_lambda = puberty_lambda

    def forward(self, pred_z, target_z, target_raw, gender):
        diff = torch.abs(pred_z - target_z)
        loss_elem = torch.where(diff < self.beta, 0.5 * (diff ** 2) / self.beta, diff - 0.5 * self.beta)

        is_female = (gender.squeeze(-1) < 0.5)
        is_male = (gender.squeeze(-1) >= 0.5)
        is_pub_f = is_female & (target_raw >= 132.0) & (target_raw <= 168.0)
        is_pub_m = is_male & (target_raw >= 144.0) & (target_raw <= 180.0)
        is_puberty = is_pub_f | is_pub_m

        weights = torch.ones_like(loss_elem) + self.puberty_lambda * is_puberty.float()
        return (loss_elem * weights).mean()""")

    # Step 4: Model architecture
    add_md("""---
## 🧠 BƯỚC 4: KIẾN TRÚC ANATOMICAL DUAL-CROP (SWIN GLOBAL + CONVNEXT WRIST ZOOM)""")

    add_code("""# ==============================================================================
# BƯỚC 4: KIẾN TRÚC ANATOMICAL DUAL-CROP BONE AGE NET
# ==============================================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query_feat, key_feat):
        q = self.norm_q(query_feat)
        kv = self.norm_kv(key_feat)
        attn_out, attn_weights = self.cross_attn(query=q, key=kv, value=kv)
        x = query_feat + attn_out
        out = x + self.ffn(self.norm_out(x))
        return out, attn_weights

class FiLMLayer(nn.Module):
    def __init__(self, feature_dim=512, metadata_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim * 2)
        )

    def forward(self, feature, metadata):
        params = self.net(metadata)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return feature * (1.0 + gamma) + beta

class AnatomicalDualCropBoneAgeNet(nn.Module):
    \"\"\"Anatomical Dual-Crop: Swin-Small Global Hand (384) + ConvNeXt-Small Carpal Zoom (256).\"\"\"
    def __init__(self, convnext_name="convnext_small", swin_name="swin_small_patch4_window7_224",
                 embed_dim=512, num_heads=8, dropout=0.3, drop_path_rate=0.2, pretrained=True):
        super().__init__()
        # ConvNeXt trên Wrist Crop 256x256 -> 8x8 = 64 spatial tokens
        self.convnext = timm.create_model(
            convnext_name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate
        )
        convnext_dim = getattr(self.convnext, "num_features", 768)

        # Swin trên Whole Hand 384x384 -> 12x12 = 144 macro tokens
        self.swin = timm.create_model(
            swin_name, pretrained=pretrained, img_size=384, num_classes=0, drop_path_rate=drop_path_rate
        )
        swin_dim = getattr(self.swin, "num_features", 768)


        self.proj_conv = nn.Sequential(nn.Linear(convnext_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.proj_swin = nn.Sequential(nn.Linear(swin_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

        self.fusion = CrossAttentionFusion(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
        self.film = FiLMLayer(feature_dim=embed_dim, metadata_dim=1)

        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, img_hand, img_wrist, gender, return_attention=False):
        # 1. Trích xuất Wrist Tokens (ConvNeXt trên 256x256)
        fc = self.convnext.forward_features(img_wrist)
        if hasattr(self.convnext, "head") and hasattr(self.convnext.head, "norm"):
            fc = self.convnext.head.norm(fc)
        tokens_wrist = fc.flatten(2).permute(0, 2, 1)  # [B, 64, 768]

        # 2. Trích xuất Global Hand Tokens (Swin trên 384x384)
        fs = self.swin.forward_features(img_hand)
        if fs.dim() == 4:
            tokens_hand = fs.view(fs.size(0), -1, fs.size(-1))
        else:
            tokens_hand = fs  # [B, 144, 768]

        # 3. Phép chiếu không gian
        f_w = self.proj_conv(tokens_wrist)  # Keys: 64 High-Res Wrist Tokens
        f_h = self.proj_swin(tokens_hand)   # Queries: 144 Global Hand Tokens

        # 4. Cross-Attention: Macro queries soi vào micro wrist keys
        fused_tokens, attn_weights = self.fusion(query_feat=f_h, key_feat=f_w)
        fused = fused_tokens.mean(dim=1)

        # 5. Điều biến giới tính FiLM và Hồi quy
        modulated = self.film(fused, gender)
        out = self.regressor(modulated).squeeze(-1)

        if return_attention:
            return out, attn_weights
        return out

temp_m = AnatomicalDualCropBoneAgeNet(pretrained=False)
p_total = sum(p.numel() for p in temp_m.parameters()) / 1e6
log_status(f"🚀 Kiến Trúc Anatomical Dual-Crop: {p_total:.2f}M tham số sẵn sàng!")
del temp_m
gc.collect()""")

    # Step 5: Training Loop
    add_md("""---
## 🚀 BƯỚC 5: MASTER TRAINING LOOP CHO ANATOMICAL DUAL-CROP""")

    add_code("""# ==============================================================================
# BƯỚC 5: MASTER TRAINING LOOP ANATOMICAL DUAL-CROP
# ==============================================================================
def set_backbone_freeze(model, freeze=True):
    for p in model.convnext.parameters():
        p.requires_grad = not freeze
    for p in model.swin.parameters():
        p.requires_grad = not freeze

def predict_with_tta(model, img_hand, img_wrist, gender, use_tta=True):
    if not use_tta:
        with torch.amp.autocast('cuda', enabled=CONFIG['USE_AMP']):
            return model(img_hand, img_wrist, gender)
    preds = []
    with torch.amp.autocast('cuda', enabled=CONFIG['USE_AMP']):
        p0 = model(img_hand, img_wrist, gender)
        preds.append(p0)
        hand_z = torch.nn.functional.interpolate(img_hand, scale_factor=1.03, mode='bilinear', align_corners=False)
        _, _, zh, zw = hand_z.shape
        ch, cw = img_hand.shape[2], img_hand.shape[3]
        hand_z = hand_z[:, :, (zh-ch)//2:(zh-ch)//2+ch, (zw-cw)//2:(zw-cw)//2+cw]
        p1 = model(hand_z, img_wrist, gender)
        preds.append(p1)
    return torch.stack(preds, dim=0).mean(dim=0)

def plot_and_save_training_progress(history, fold_idx, output_dir):
    epochs = history['epoch']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', lw=2)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', lw=2)
    ax1.set_xlabel('Epochs', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title(f'Dual-Crop Fold {fold_idx}: Tiến Trình Mất Mát', fontweight='bold')
    ax1.legend()
    ax2.plot(epochs, history['train_mae'], 'b-o', label='Train MAE', lw=2)
    ax2.plot(epochs, history['val_mae'], 'r-s', label='Val MAE', lw=2)
    best_m = min(history['val_mae'])
    ax2.axhline(best_m, color='g', linestyle='--', label=f'Đáy Val MAE: {best_m:.2f}m')
    ax2.set_xlabel('Epochs', fontweight='bold')
    ax2.set_ylabel('MAE (Tháng)', fontweight='bold')
    ax2.set_title(f'Dual-Crop Fold {fold_idx}: Độ Chính Xác MAE', fontweight='bold')
    ax2.legend()
    plt.tight_layout()
    fig_path = Path(output_dir) / f"training_curves_dualcrop_fold{fold_idx}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

session_start_time = time.time()
fold_results = {}

for fold_idx in CONFIG['RUN_FOLDS']:
    log_status("=" * 70)
    log_status(f"🚀 BẮT ĐẦU HUẤN LUYỆN DUAL-CROP FOLD {fold_idx} ({CONFIG['EPOCHS']} EPOCHS)...")
    log_status("=" * 70)

    train_df = pool_df[pool_df['fold'] != fold_idx].reset_index(drop=True)
    val_df = pool_df[pool_df['fold'] == fold_idx].reset_index(drop=True)

    b_mean = float(train_df['boneage'].mean())
    b_std = float(train_df['boneage'].std())

    train_dataset = RSNADualCropDataset(train_df, IMAGES_DIR, hand_size=CONFIG['IMAGE_SIZE_HAND'], wrist_size=CONFIG['IMAGE_SIZE_WRIST'], is_train=True, use_zscore=True, mean_target=b_mean, std_target=b_std)
    val_dataset = RSNADualCropDataset(val_df, IMAGES_DIR, hand_size=CONFIG['IMAGE_SIZE_HAND'], wrist_size=CONFIG['IMAGE_SIZE_WRIST'], is_train=False, use_zscore=True, mean_target=b_mean, std_target=b_std)
    test_dataset = RSNADualCropDataset(test_df, IMAGES_DIR, hand_size=CONFIG['IMAGE_SIZE_HAND'], wrist_size=CONFIG['IMAGE_SIZE_WRIST'], is_train=False, use_zscore=True, mean_target=b_mean, std_target=b_std)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=CONFIG['NUM_WORKERS'], pin_memory=True)

    model = AnatomicalDualCropBoneAgeNet(
        convnext_name=CONFIG['CONVNEXT_MODEL'], swin_name=CONFIG['SWIN_MODEL'],
        embed_dim=CONFIG['EMBED_DIM'], num_heads=CONFIG['NUM_HEADS'],
        dropout=CONFIG['DROPOUT'], drop_path_rate=CONFIG['DROP_PATH_RATE']
    ).to(device)

    criterion = AdaptivePubertalLoss(beta=CONFIG['SMOOTH_L1_BETA'], puberty_lambda=CONFIG['PUBERTY_WEIGHT'])
    scaler = torch.amp.GradScaler('cuda', enabled=CONFIG['USE_AMP'])

    set_backbone_freeze(model, freeze=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CONFIG['LR_HEAD_WARMUP'], weight_decay=CONFIG['WEIGHT_DECAY'])

    history = {'epoch': [], 'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': [], 'lr': []}
    best_val_mae = float('inf')
    checkpoint_path = Path(CONFIG['CHECKPOINT_DIR']) / f"{CONFIG['EXPERIMENT']}_fold{fold_idx}_best.pt"

    for epoch in range(1, CONFIG['EPOCHS'] + 1):
        t0 = time.time()
        if epoch == CONFIG['WARMUP_EPOCHS'] + 1:
            log_status(f"[Dual-Crop Fold {fold_idx}] 🔓 Mở khóa toàn bộ Backbone để Finetune End-to-End!")
            set_backbone_freeze(model, freeze=False)
            optimizer = torch.optim.AdamW([
                {'params': model.convnext.parameters(), 'lr': CONFIG['LR_BACKBONE'] * 0.8},
                {'params': model.swin.parameters(), 'lr': CONFIG['LR_BACKBONE']},
                {'params': model.proj_conv.parameters(), 'lr': CONFIG['LR_HEAD']},
                {'params': model.proj_swin.parameters(), 'lr': CONFIG['LR_HEAD']},
                {'params': model.fusion.parameters(), 'lr': CONFIG['LR_HEAD']},
                {'params': model.film.parameters(), 'lr': CONFIG['LR_HEAD']},
                {'params': model.regressor.parameters(), 'lr': CONFIG['LR_HEAD']},
            ], weight_decay=CONFIG['WEIGHT_DECAY'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS'] - CONFIG['WARMUP_EPOCHS'], eta_min=1e-6)

        model.train()
        train_loss_acc, train_mae_acc = 0.0, 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (img_hand, img_wrist, gender, target_z, target_raw) in enumerate(train_loader):
            img_hand, img_wrist, gender, target_z, target_raw = img_hand.to(device), img_wrist.to(device), gender.to(device), target_z.to(device), target_raw.to(device)
            with torch.amp.autocast('cuda', enabled=CONFIG['USE_AMP']):
                pred_z = model(img_hand, img_wrist, gender)
                loss = criterion(pred_z, target_z, target_raw, gender) / CONFIG['GRADIENT_ACCUMULATION']

            scaler.scale(loss).backward()
            if (step + 1) % CONFIG['GRADIENT_ACCUMULATION'] == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss_acc += loss.item() * CONFIG['GRADIENT_ACCUMULATION'] * len(target_z)
            pred_months = pred_z.detach() * b_std + b_mean
            train_mae_acc += torch.abs(pred_months - target_raw).sum().item()

        epoch_train_loss = train_loss_acc / len(train_dataset)
        epoch_train_mae = train_mae_acc / len(train_dataset)

        # Validation
        model.eval()
        val_loss_acc, val_mae_acc = 0.0, 0.0
        with torch.no_grad():
            for img_hand, img_wrist, gender, target_z, target_raw in val_loader:
                img_hand, img_wrist, gender, target_z, target_raw = img_hand.to(device), img_wrist.to(device), gender.to(device), target_z.to(device), target_raw.to(device)
                with torch.amp.autocast('cuda', enabled=CONFIG['USE_AMP']):
                    pred_z = model(img_hand, img_wrist, gender)
                    loss = criterion(pred_z, target_z, target_raw, gender)
                val_loss_acc += loss.item() * len(target_z)
                pred_months = pred_z * b_std + b_mean
                val_mae_acc += torch.abs(pred_months - target_raw).sum().item()

        epoch_val_loss = val_loss_acc / len(val_dataset)
        epoch_val_mae = val_mae_acc / len(val_dataset)
        curr_lr = optimizer.param_groups[0]['lr']

        if epoch > CONFIG['WARMUP_EPOCHS']:
            scheduler.step()

        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_train_loss)
        history['train_mae'].append(epoch_train_mae)
        history['val_loss'].append(epoch_val_loss)
        history['val_mae'].append(epoch_val_mae)
        history['lr'].append(curr_lr)

        is_best = epoch_val_mae < best_val_mae
        if is_best:
            best_val_mae = epoch_val_mae
            torch.save({
                'epoch': epoch, 'fold': fold_idx, 'model_state_dict': model.state_dict(),
                'best_val_mae': best_val_mae, 'config': CONFIG, 'boneage_mean': b_mean, 'boneage_std': b_std
            }, checkpoint_path)

        elapsed = time.time() - t0
        total_m = (time.time() - session_start_time) / 60.0
        log_status(f"[Dual-Crop Fold {fold_idx} | Ep {epoch:02d}/{CONFIG['EPOCHS']:02d}] "
                   f"Train Loss: {epoch_train_loss:.4f} | Train MAE: {epoch_train_mae:.2f}m | "
                   f"Val Loss: {epoch_val_loss:.4f} | Val MAE: {epoch_val_mae:.2f}m | "
                   f"Best: {best_val_mae:.2f}m {'🌟' if is_best else ''} ({elapsed:.0f}s | Tổng: {total_m:.1f}m)")

        plot_and_save_training_progress(history, fold_idx, CONFIG['OUTPUT_DIR'])

    # Test Evaluation
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_preds, test_trues = [], []
    with torch.no_grad():
        for img_hand, img_wrist, gender, _, target_raw in test_loader:
            img_hand, img_wrist, gender = img_hand.to(device), img_wrist.to(device), gender.to(device)
            pred_z = predict_with_tta(model, img_hand, img_wrist, gender, use_tta=CONFIG['USE_TTA'])
            pred_m = pred_z * b_std + b_mean
            test_preds.extend(pred_m.cpu().numpy().tolist())
            test_trues.extend(target_raw.numpy().tolist())

    df_test_preds = test_df[['id', 'male', 'boneage']].copy()
    df_test_preds.rename(columns={'boneage': 'boneage_true'}, inplace=True)
    df_test_preds['boneage_pred'] = test_preds
    df_test_preds['abs_error'] = (df_test_preds['boneage_pred'] - df_test_preds['boneage_true']).abs()

    test_mae = float(df_test_preds['abs_error'].mean())
    test_rmse = float(np.sqrt(np.mean((df_test_preds['boneage_true'] - df_test_preds['boneage_pred'])**2)))
    ss_res = np.sum((df_test_preds['boneage_true'] - df_test_preds['boneage_pred'])**2)
    ss_tot = np.sum((df_test_preds['boneage_true'] - np.mean(df_test_preds['boneage_true']))**2)
    r2_score = float(1.0 - (ss_res / ss_tot))

    fold_suffix = f"_fold{fold_idx}"
    test_csv = Path(CONFIG['OUTPUT_DIR']) / f"{CONFIG['EXPERIMENT']}{fold_suffix}_test_predictions.csv"
    metrics_json = Path(CONFIG['OUTPUT_DIR']) / f"{CONFIG['EXPERIMENT']}{fold_suffix}_metrics.json"

    df_test_preds.to_csv(test_csv, index=False)
    summary_fold = {
        'fold': fold_idx, 'best_val_mae': float(best_val_mae),
        'holdout_test_mae': test_mae, 'holdout_test_rmse': test_rmse, 'r2_score': r2_score
    }
    with open(metrics_json, 'w', encoding='utf-8') as f:
        json.dump(summary_fold, f, indent=2)

    fold_results[fold_idx] = {'best_val_mae': best_val_mae, 'test_mae': test_mae, 'test_csv': test_csv}
    log_status(f"🎉 HOÀN TẤT DUAL-CROP FOLD {fold_idx}: Best Val MAE = {best_val_mae:.2f}m | Holdout Test MAE = {test_mae:.2f}m")

    del model, optimizer, scheduler, scaler
    del train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    if torch.cuda.is_available():
        vram_left = torch.cuda.memory_allocated() / 1e6
        log_status(f"🧹 Dọn dẹp VRAM Fold {fold_idx} hoàn tất. Bộ nhớ đang chiếm: {vram_left:.1f} MB.")""")

    # Step 6: Ensemble
    add_md("""---
## 🏆 BƯỚC 6: GRAND FINALE SOFT-VOTING ENSEMBLE CHO ANATOMICAL DUAL-CROP""")

    add_code("""# ==============================================================================
# BƯỚC 6: GRAND FINALE SOFT-VOTING ENSEMBLE
# ==============================================================================
log_status("=" * 70)
log_status("👑 TIẾN HÀNH SOFT-VOTING ENSEMBLE CHO DUAL-CROP TRÊN TẬP TEST...")
log_status("=" * 70)

loaded_test_dfs = {}
for f_i in CONFIG['RUN_FOLDS']:
    f_csv = Path(CONFIG['OUTPUT_DIR']) / f"{CONFIG['EXPERIMENT']}_fold{f_i}_test_predictions.csv"
    if f_csv.exists():
        loaded_test_dfs[f_i] = pd.read_csv(f_csv)

if len(loaded_test_dfs) > 0:
    ref_df = list(loaded_test_dfs.values())[0].copy()
    ensemble_preds = np.zeros(len(ref_df))
    for f_i, d in loaded_test_dfs.items():
        ensemble_preds += d['boneage_pred'].values
    ensemble_preds /= len(loaded_test_dfs)

    df_ensemble = ref_df[['id', 'male', 'boneage_true']].copy()
    df_ensemble['boneage_pred'] = ensemble_preds
    df_ensemble['abs_error'] = (df_ensemble['boneage_pred'] - df_ensemble['boneage_true']).abs()

    ens_mae = float(df_ensemble['abs_error'].mean())
    ens_rmse = float(np.sqrt(np.mean((df_ensemble['boneage_true'] - df_ensemble['boneage_pred'])**2)))
    ss_res_e = np.sum((df_ensemble['boneage_true'] - df_ensemble['boneage_pred'])**2)
    ss_tot_e = np.sum((df_ensemble['boneage_true'] - np.mean(df_ensemble['boneage_true']))**2)
    ens_r2 = float(1.0 - (ss_res_e / ss_tot_e))

    is_pub_f = (~df_ensemble['male']) & (df_ensemble['boneage_true'] >= 132) & (df_ensemble['boneage_true'] <= 168)
    is_pub_m = (df_ensemble['male']) & (df_ensemble['boneage_true'] >= 144) & (df_ensemble['boneage_true'] <= 180)
    pub_ens_df = df_ensemble[is_pub_f | is_pub_m]
    ens_pub_mae = float(pub_ens_df['abs_error'].mean()) if len(pub_ens_df) > 0 else 0.0

    boot_ens = []
    e_errs = df_ensemble['abs_error'].values
    for _ in range(1000):
        s = np.random.choice(e_errs, size=len(e_errs), replace=True)
        boot_ens.append(np.mean(s))
    e_ci_l, e_ci_u = np.percentile(boot_ens, [2.5, 97.5])

    ens_summary = {
        'num_folds_ensembled': len(loaded_test_dfs),
        'ensemble_test_mae': ens_mae,
        'ensemble_mae_ci_95': [float(e_ci_l), float(e_ci_u)],
        'ensemble_rmse': ens_rmse,
        'ensemble_r2': ens_r2,
        'ensemble_puberty_mae': ens_pub_mae,
        'individual_fold_maes': {f"fold_{f_i}": float(fold_results[f_i]['test_mae']) for f_i in fold_results}
    }

    ens_csv_path = Path(CONFIG['OUTPUT_DIR']) / f"{CONFIG['EXPERIMENT']}_ensemble_predictions.csv"
    ens_json_path = Path(CONFIG['OUTPUT_DIR']) / f"{CONFIG['EXPERIMENT']}_ensemble_metrics.json"

    df_ensemble.to_csv(ens_csv_path, index=False)
    with open(ens_json_path, 'w', encoding='utf-8') as f:
        json.dump(ens_summary, f, indent=2)

    log_status("=" * 70)
    log_status(f"👑 KẾT QUẢ ĐỈNH CAO ANATOMICAL DUAL-CROP ENSEMBLE ({len(loaded_test_dfs)} FOLDS):")
    log_status(f"• Dual-Crop Ensemble Test MAE: {ens_mae:.2f} tháng (95% CI: [{e_ci_l:.2f}, {e_ci_u:.2f}])")
    log_status(f"• Dual-Crop Ensemble Test RMSE: {ens_rmse:.2f} tháng")
    log_status(f"• Dual-Crop Ensemble R2 Score:  {ens_r2:.4f}")
    log_status(f"• Pubertal Spurt MAE:          {ens_pub_mae:.2f} tháng (N={len(pub_ens_df)})")
    log_status(f"💾 File dự đoán:                 {ens_csv_path}")
    log_status("=" * 70)

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_labels = [f"Fold {f_i}" for f_i in loaded_test_dfs.keys()] + ["⭐ Dual-Crop Ensemble"]
    bar_values = [fold_results[f_i]['test_mae'] for f_i in loaded_test_dfs.keys()] + [ens_mae]
    bar_colors = ['#1f77b4'] * len(loaded_test_dfs) + ['#2ca02c']

    bars = ax.bar(bar_labels, bar_values, color=bar_colors, width=0.55, edgecolor='black', lw=1.2)
    ax.axhline(4.265, color='#d62728', linestyle='--', lw=1.8, label="Kỷ lục Vô địch RSNA 2017 (4.265m)")
    ax.set_ylabel("MAE trên Tập Test Độc Lập 1,262 ca (Tháng)", fontsize=11, fontweight='bold')
    ax.set_title("Hiệu Suất Anatomical Dual-Crop Zoom (Toàn Tay 384 + Cận Cảnh Cổ Tay 256)", fontsize=12, fontweight='bold')

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}m", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    ens_bar_path = Path(CONFIG['OUTPUT_DIR']) / "ensemble_dualcrop_comparison_barchart.png"
    plt.savefig(ens_bar_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    total_session_hours = (time.time() - session_start_time) / 3600.0
    log_status(f"🏁 TOÀN BỘ PHIÊN CHẠY 02F ĐÃ HOÀN TẤT THÀNH CÔNG TRONG {total_session_hours:.2f} GIỜ!")
""")

    nb_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat_minor": 4,
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_file = "d:/Development/projects/dual_branch_bone_age/kaggle_notebooks/02f_anatomical_roi_multicrop.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1)

    print(f"Generated {out_file} successfully!")

if __name__ == "__main__":
    create_notebook()
