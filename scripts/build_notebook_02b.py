"""Script to generate Notebook 02b: 02b_dual_branch_cross_attention_film.ipynb."""

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
    add_md("""# 🧬 MỐC 2: BƯỚC NHẢY VỌT KIẾN TRÚC DUAL-BRANCH (CONVNEXT + SWINT + CROSS-ATTENTION + FILM)
### Đề tài: *Harmonizing Local Inductive Bias and Global Contextual Reasoning: A Dual-Branch ConvNeXt-Swin Transformer Architecture with Gender-Modulated Cross-Attention for Pediatric Bone Age Assessment*

---
## 🎯 Mục Tiêu Khoa Học & Ý Nghĩa Câu Chuyện (Milestone 2):
- **Giải quyết trực diện điểm nghẽn của Inception-v3:** Dung hòa giữa **tính quy nạp cục bộ (CNN Inductive Bias)** và **khả năng mô hình hóa ngữ cảnh tầm xa (Global Context Modeling)**.
- **Nhánh Vi mô - ConvNeXt-Small ($384\\times384$):** Bắt chính xác độ khép của đĩa sụn tiếp hợp (epiphyseal plate) và độ dày vệt vôi hóa bằng $7\\times7$ depthwise separable convolutions.
- **Nhánh Vĩ mô - Swin Transformer-Small ($224\\times224$):** Tính toán ma trận tương quan giữa 8 xương cổ tay và 5 tia ngón tay bằng cơ chế *Shifted Window Self-Attention* độ phức tạp $O(N)$ theo chuẩn Tanner-Whitehouse (TW3).
- **Bộ tương tác đa phương thức (Cross-Attention & FiLM):** Swin tokens (Query) hội thoại trực tiếp với ConvNeXt feature map (Key/Value). Sau đó được điều hòa thích ứng theo giới tính bằng lớp **FiLM (Feature-wise Linear Modulation)** thay vì ghép nối thô (naive concatenation).
- **Chiến lược tối ưu 2 giai đoạn (Gradual Unfreezing):** Đóng băng backbones trong pha Warmup, sau đó mở khóa toàn bộ với tốc độ học phân biệt (Discriminative Learning Rates) và Cosine Annealing.""")

    add_code("""# Cài đặt thư viện phụ trợ (timm, albumentations)
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
import timm

# Thiết lập hiển thị đồ họa
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# BẢNG CẤU HÌNH THÍ NGHIỆM EXP 4 (DUAL-BRANCH HYBRID NET)
CONFIG = {
    'EXPERIMENT': 'exp4_dual_branch_cross_film',
    'SEED': 42,
    'CONVNEXT_MODEL': 'convnext_small.fb_in22k_ft_in1k_384',
    'SWIN_MODEL': 'swin_small_patch4_window7_224.ms_in22k_ft_in1k',
    'IMAGE_SIZE_CONV': 384,     # Độ phân giải nhánh vi mô ConvNeXt
    'IMAGE_SIZE_SWIN': 224,     # Độ phân giải nhánh vĩ mô Swin Transformer
    'EMBED_DIM': 512,
    'NUM_HEADS': 8,
    'DROPOUT': 0.3,
    'BATCH_SIZE': 16,
    'NUM_WORKERS': 4,
    'EPOCHS': 20,
    'WARMUP_EPOCHS': 2,         # Đóng băng backbones trong 2 epoch đầu
    'LR_HEAD_WARMUP': 1e-3,     # Tốc độ học pha warmup
    'LR_BACKBONE': 5e-5,        # Tốc độ học backbone khi finetune
    'LR_HEAD': 2e-4,            # Tốc độ học head khi finetune
    'WEIGHT_DECAY': 1e-4,
    'GRADIENT_ACCUMULATION': 2, # Effective Batch Size = 32
    'USE_AMP': True,
    'CHECKPOINT_DIR': '/kaggle/working/checkpoints',
    'OUTPUT_DIR': '/kaggle/working/outputs',
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
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)""")

    # Data Loader Auto-discovery
    add_md("""---
## 📂 BƯỚC 2: QUÉT TỰ ĐỘNG DỮ LIỆU ĐÃ TIỀN XỬ LÝ (TỪ NOTEBOOK 01)
Nạp file nhãn phân tầng `train_stratified.csv` và thư mục ảnh $512\\times512$.""")

    add_code("""# ==============================================================================
# BƯỚC 2: TỰ ĐỘNG KẾT NỐI DATASET TIỀN XỬ LÝ
# ==============================================================================
DATASET_DIR = None
STRATIFIED_CSV_PATH = None
IMAGES_DIR = None

for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f == 'train_stratified.csv':
            STRATIFIED_CSV_PATH = Path(root) / f
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
print(f"✅ Đã kết nối Dataset thành công:")
print(f"• File phân tầng: {STRATIFIED_CSV_PATH}")
print(f"• Thư mục ảnh:    {IMAGES_DIR}")

df_all = pd.read_csv(STRATIFIED_CSV_PATH)
train_df = df_all[df_all['split'] == 'train'].reset_index(drop=True)
val_df = df_all[df_all['split'] == 'val'].reset_index(drop=True)
test_df = df_all[df_all['split'] == 'test'].reset_index(drop=True)

print(f"📊 Phân bổ mẫu: Train = {len(train_df):,} | Val = {len(val_df):,} | Test = {len(test_df):,}")""")

    # Dual-Resolution Dataset
    add_md("""---
## 📦 BƯỚC 3: DUAL-RESOLUTION DATASET ĐA PHƯƠNG THỨC
Xây dựng Dataset trích xuất đồng thời 2 độ phân giải:
- **`img_conv` ($384\\times384$):** Cung cấp trường thị giác sắc nét cho mạng CNN bắt viền sụn.
- **`img_swin` ($224\\times224$):** Cung cấp kích thước chuẩn cho các patch và window $7\\times7$ của Swin Transformer.""")

    add_code("""# ==============================================================================
# BƯỚC 3: XÂY DỰNG DATASET HỖ TRỢ ĐỒNG THỜI HAI ĐỘ PHÂN GIẢI
# ==============================================================================
class RSNADualBranchDataset(Dataset):
    def __init__(self, df, img_dir, conv_size=384, swin_size=224, is_train=True):
        self.df = df
        self.img_dir = img_dir
        self.conv_size = conv_size
        self.swin_size = swin_size
        self.is_train = is_train

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def _normalize(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return (tensor - self.mean) / self.std

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = int(row['id'])
        img_path = self.img_dir / f"{img_id}.png"

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((512, 512, 3), dtype=np.uint8)

        # Tạo ảnh nhánh ConvNeXt (384x384)
        img_conv = cv2.resize(img, (self.conv_size, self.conv_size), interpolation=cv2.INTER_AREA)
        # Tạo ảnh nhánh Swin Transformer (224x224)
        img_swin = cv2.resize(img, (self.swin_size, self.swin_size), interpolation=cv2.INTER_AREA)

        tensor_conv = self._normalize(img_conv)
        tensor_swin = self._normalize(img_swin)

        gender = torch.tensor([1.0 if row['male'] else 0.0], dtype=torch.float32)
        boneage = torch.tensor(row['boneage'], dtype=torch.float32)

        return tensor_conv, tensor_swin, gender, boneage

train_dataset = RSNADualBranchDataset(train_df, IMAGES_DIR, conv_size=CONFIG['IMAGE_SIZE_CONV'], swin_size=CONFIG['IMAGE_SIZE_SWIN'], is_train=True)
val_dataset = RSNADualBranchDataset(val_df, IMAGES_DIR, conv_size=CONFIG['IMAGE_SIZE_CONV'], swin_size=CONFIG['IMAGE_SIZE_SWIN'], is_train=False)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, 
                          num_workers=CONFIG['NUM_WORKERS'], pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, 
                        num_workers=CONFIG['NUM_WORKERS'], pin_memory=True)

print(f"✅ Khởi tạo DataLoader 2 nhánh thành công: {len(train_loader)} batches/epoch")""")

    # Model Architecture: ConvNeXt + Swin + Cross-Attention + FiLM
    add_md("""---
## 🧠 BƯỚC 4: ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH NHÁNH KÉP (DUAL-BRANCH HYBRID NET)
Tích hợp trực tiếp các module tiên tiến:
1. **Lớp FiLM (Feature-wise Linear Modulation):** Điều hòa đặc trưng theo vector giới tính thông qua $\\mathbf{h} = \\mathbf{x} \\odot (1 + \\boldsymbol{\\gamma}) + \\boldsymbol{\\beta}$ với trọng số khởi tạo ban đầu bằng 0 (Zero-initialized).
2. **Module Cross-Attention:** Swin macro representation (Query) truy vấn đặc trưng ConvNeXt micro representation (Key/Value) kèm residual connection.
3. **Mạng tích hợp `DualBranchBoneAgeNet`:** Ghép nối hoàn chỉnh hai nhánh backbone và Regression Head.""")

    add_code("""# ==============================================================================
# BƯỚC 4: ĐỊNH NGHĨA CÁC MODULE KIẾN TRÚC HYBRID
# ==============================================================================

class FiLMLayer(nn.Module):
    \"\"\"Điều hòa feature map bằng thông tin lâm sàng (giới tính sinh học)\"\"\"
    def __init__(self, feature_dim, metadata_dim=1):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.GELU(),
            nn.Linear(64, feature_dim * 2)  # Sinh gamma (scale) và beta (shift)
        )
        # Khởi tạo trọng số zero để ban đầu lớp hoạt động như Identity
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def forward(self, x, meta):
        film_params = self.generator(meta)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        return x * (1.0 + gamma) + beta


class CrossAttentionFusion(nn.Module):
    \"\"\"Module tương tác đa phương thức: Swin query truy vấn ConvNeXt keys\"\"\"
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, 
                                                dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query_feat, key_feat):
        # query_feat: [B, 1, embed_dim] (Swin macro)
        # key_feat:   [B, 1, embed_dim] (ConvNeXt micro)
        attn_out, _ = self.cross_attn(query=query_feat, key=key_feat, value=key_feat)
        fused = self.norm(query_feat + attn_out).squeeze(1)  # Residual connection
        return fused


class DualBranchBoneAgeNet(nn.Module):
    \"\"\"Kiến trúc đề xuất kết hợp ConvNeXt vi mô và Swin vĩ mô\"\"\"
    def __init__(self, convnext_name="convnext_small.fb_in22k_ft_in1k_384",
                 swin_name="swin_small_patch4_window7_224.ms_in22k_ft_in1k",
                 embed_dim=512, num_heads=8, dropout=0.3):
        super().__init__()

        # 1. Nhánh Vi mô (ConvNeXt)
        self.convnext = timm.create_model(convnext_name, pretrained=True, num_classes=0)
        convnext_dim = self.convnext.num_features

        # 2. Nhánh Vĩ mô (Swin Transformer)
        self.swin = timm.create_model(swin_name, pretrained=True, num_classes=0)
        swin_dim = self.swin.num_features

        # Chiếu về cùng không gian nhúng
        self.proj_conv = nn.Sequential(nn.Linear(convnext_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.proj_swin = nn.Sequential(nn.Linear(swin_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

        # 3. Module Cross-Attention
        self.fusion = CrossAttentionFusion(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)

        # 4. Điều hòa giới tính qua FiLM
        self.film = FiLMLayer(feature_dim=embed_dim, metadata_dim=1)

        # 5. Đầu hồi quy (Regression Head)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, img_conv, img_swin, gender):
        # Forward qua 2 nhánh
        feat_conv = self.convnext(img_conv)       # [B, convnext_dim]
        feat_swin = self.swin(img_swin)           # [B, swin_dim]

        # Chiếu embedding
        f_c = self.proj_conv(feat_conv).unsqueeze(1)  # [B, 1, embed_dim]
        f_s = self.proj_swin(feat_swin).unsqueeze(1)  # [B, 1, embed_dim]

        # Giao tiếp Cross-Attention
        fused = self.fusion(query_feat=f_s, key_feat=f_c)  # [B, embed_dim]

        # Điều hòa bằng vector giới tính qua FiLM
        modulated = self.film(fused, gender)              # [B, embed_dim]

        # Dự đoán tuổi xương
        out = self.regressor(modulated)
        return out.squeeze(-1)

model = DualBranchBoneAgeNet(
    convnext_name=CONFIG['CONVNEXT_MODEL'],
    swin_name=CONFIG['SWIN_MODEL'],
    embed_dim=CONFIG['EMBED_DIM'],
    num_heads=CONFIG['NUM_HEADS'],
    dropout=CONFIG['DROPOUT']
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Khởi tạo DualBranchBoneAgeNet thành công! Tổng tham số: {total_params/1e6:.2f}M")""")

    # Two-Stage Gradual Unfreezing Trainer
    add_md("""---
## 🔄 BƯỚC 5: TRAINING ENGINE 2 GIAI ĐOẠN (GRADUAL UNFREEZING & COSINE ANNEALING)
Chiến lược tối ưu hóa chuẩn nghiên cứu:
- **Pha 1 (Warmup):** Đóng băng cả 2 backbone `convnext` và `swin` trong 2 epoch đầu. Chỉ huấn luyện các lớp chiếu, fusion, film và head với learning rate cao ($1\\text{e-}3$).
- **Pha 2 (Fine-tuning):** Mở khóa toàn bộ mô hình, phân bổ Discriminative Learning Rates ($5\\text{e-}5$ cho backbones và $2\\text{e-}4$ cho heads) kết hợp Cosine Annealing.""")

    add_code("""# ==============================================================================
# BƯỚC 5: HUẤN LUYỆN 2 GIAI ĐOẠN VỚI GRADUAL UNFREEZING
# ==============================================================================
criterion = nn.SmoothL1Loss(beta=1.0)
scaler = GradScaler(enabled=CONFIG['USE_AMP'])

def set_backbone_freeze(model, freeze=True):
    for p in model.convnext.parameters():
        p.requires_grad = not freeze
    for p in model.swin.parameters():
        p.requires_grad = not freeze

# Pha 1: Đóng băng backbones
set_backbone_freeze(model, freeze=True)
print("🔒 Pha 1: Đã đóng băng trọng số của cả 2 Backbone ConvNeXt và Swin!")

# Khởi tạo optimizer cho pha 1
trainable_params_phase1 = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params_phase1, lr=CONFIG['LR_HEAD_WARMUP'], weight_decay=CONFIG['WEIGHT_DECAY'])

history = {'epoch': [], 'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': [], 'lr': []}
best_val_mae = float('inf')
checkpoint_path = Path(CONFIG['CHECKPOINT_DIR']) / f"{CONFIG['EXPERIMENT']}_best.pt"

print(f"🚀 BẮT ĐẦU HUẤN LUYỆN DUAL-BRANCH ({CONFIG['EPOCHS']} EPOCHS)...")

for epoch in range(1, CONFIG['EPOCHS'] + 1):
    start_time = time.time()

    # Chuyển đổi sang Pha 2 khi hết số epoch Warmup
    if epoch == CONFIG['WARMUP_EPOCHS'] + 1:
        print("🔓 Pha 2: MỞ KHÓA TOÀN BỘ BACKBONES ĐỂ FINETUNE END-TO-END!")
        set_backbone_freeze(model, freeze=False)
        # Phân bổ Discriminative Learning Rates
        optimizer = torch.optim.AdamW([
            {'params': model.convnext.parameters(), 'lr': CONFIG['LR_BACKBONE']},
            {'params': model.swin.parameters(), 'lr': CONFIG['LR_BACKBONE']},
            {'params': model.proj_conv.parameters(), 'lr': CONFIG['LR_HEAD']},
            {'params': model.proj_swin.parameters(), 'lr': CONFIG['LR_HEAD']},
            {'params': model.fusion.parameters(), 'lr': CONFIG['LR_HEAD']},
            {'params': model.film.parameters(), 'lr': CONFIG['LR_HEAD']},
            {'params': model.regressor.parameters(), 'lr': CONFIG['LR_HEAD']},
        ], weight_decay=CONFIG['WEIGHT_DECAY'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG['EPOCHS'] - CONFIG['WARMUP_EPOCHS'], eta_min=1e-6
        )

    model.train()
    total_train_loss, total_train_mae, train_samples = 0.0, 0.0, 0
    optimizer.zero_grad()

    for step, (img_conv, img_swin, genders, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch:02d}/{CONFIG['EPOCHS']:02d} [Train]")):
        img_conv = img_conv.to(device, non_blocking=True)
        img_swin = img_swin.to(device, non_blocking=True)
        genders = genders.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=CONFIG['USE_AMP']):
            preds = model(img_conv, img_swin, genders)
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

    if epoch > CONFIG['WARMUP_EPOCHS']:
        scheduler.step()

    epoch_train_loss = total_train_loss / train_samples
    epoch_train_mae = total_train_mae / train_samples

    # Đánh giá kiểm định
    model.eval()
    total_val_loss, total_val_mae, val_samples = 0.0, 0.0, 0

    with torch.no_grad():
        for img_conv, img_swin, genders, targets in val_loader:
            img_conv, img_swin, genders, targets = img_conv.to(device), img_swin.to(device), genders.to(device), targets.to(device)
            with autocast(enabled=CONFIG['USE_AMP']):
                preds = model(img_conv, img_swin, genders)
                loss = criterion(preds, targets)

            batch_sz = targets.size(0)
            total_val_loss += loss.item() * batch_sz
            total_val_mae += torch.sum(torch.abs(preds - targets)).item()
            val_samples += batch_sz

    epoch_val_loss = total_val_loss / val_samples
    epoch_val_mae = total_val_mae / val_samples
    elapsed = time.time() - start_time
    curr_lr = optimizer.param_groups[-1]['lr']

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
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_val_mae': best_val_mae,
            'config': CONFIG
        }, checkpoint_path)

    print(f"Epoch [{epoch:02d}/{CONFIG['EPOCHS']:02d}] "
          f"Train Loss: {epoch_train_loss:.4f} | Train MAE: {epoch_train_mae:.2f}m | "
          f"Val Loss: {epoch_val_loss:.4f} | Val MAE: {epoch_val_mae:.2f}m | "
          f"Best Val MAE: {best_val_mae:.2f}m {'🌟' if is_best else ''} ({elapsed:.1f}s)")

    # Vẽ biểu đồ thu nhỏ trực tiếp
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot(history['epoch'], history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['epoch'], history['val_loss'], label='Val Loss', color='red', linestyle='--')
    axes[0].set_title("Dual-Branch Loss (Smooth L1)")
    axes[0].legend()
    
    axes[1].plot(history['epoch'], history['train_mae'], label='Train MAE', color='blue')
    axes[1].plot(history['epoch'], history['val_mae'], label='Val MAE', color='red', linestyle='--')
    axes[1].axhline(best_val_mae, color='green', linestyle=':', label=f"Best: {best_val_mae:.2f}m")
    axes[1].set_title("Mean Absolute Error (Months)")
    axes[1].legend()
    plt.tight_layout()
    plt.show()""")

    # Evaluation & Export
    add_md("""---
## 📊 BƯỚC 6: XUẤT TOÀN BỘ KẾT QUẢ THẬT (PREDICTIONS, LOGS & METRICS)
Nạp checkpoint tối ưu để đánh giá và xuất các file khoa học.""")

    add_code("""# ==============================================================================
# BƯỚC 6: ĐÁNH GIÁ VÀ XUẤT FILE KẾT QUẢ
# ==============================================================================
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ Đã nạp thành công checkpoint từ Epoch {checkpoint['epoch']} với Val MAE: {checkpoint['best_val_mae']:.2f} tháng")

val_results = []
with torch.no_grad():
    for idx in range(len(val_df)):
        row = val_df.iloc[idx]
        img_conv, img_swin, gender, boneage = val_dataset[idx]
        img_conv = img_conv.unsqueeze(0).to(device)
        img_swin = img_swin.unsqueeze(0).to(device)
        gender = gender.unsqueeze(0).to(device)

        with autocast(enabled=CONFIG['USE_AMP']):
            pred = model(img_conv, img_swin, gender).item()

        val_results.append({
            'id': int(row['id']),
            'male': bool(row['male']),
            'boneage_true': float(boneage.item()),
            'boneage_pred': float(pred),
            'abs_error': abs(float(pred) - float(boneage.item()))
        })

df_preds = pd.DataFrame(val_results)

errors = df_preds['abs_error'].values
overall_mae = np.mean(errors)
overall_rmse = np.sqrt(np.mean((df_preds['boneage_true'] - df_preds['boneage_pred'])**2))
ss_res = np.sum((df_preds['boneage_true'] - df_preds['boneage_pred'])**2)
ss_tot = np.sum((df_preds['boneage_true'] - np.mean(df_preds['boneage_true']))**2)
r2_score = 1.0 - (ss_res / ss_tot)

is_puberty_female = (~df_preds['male']) & (df_preds['boneage_true'] >= 132) & (df_preds['boneage_true'] <= 168)
is_puberty_male = (df_preds['male']) & (df_preds['boneage_true'] >= 144) & (df_preds['boneage_true'] <= 180)
puberty_df = df_preds[is_puberty_female | is_puberty_male]
puberty_mae = puberty_df['abs_error'].mean() if len(puberty_df) > 0 else 0.0

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

history_csv_path = Path(CONFIG['OUTPUT_DIR']) / f"{CONFIG['EXPERIMENT']}_history.csv"
preds_csv_path = Path(CONFIG['OUTPUT_DIR']) / f"{CONFIG['EXPERIMENT']}_val_predictions.csv"
metrics_json_path = Path(CONFIG['OUTPUT_DIR']) / f"{CONFIG['EXPERIMENT']}_metrics.json"

pd.DataFrame(history).to_csv(history_csv_path, index=False)
df_preds.to_csv(preds_csv_path, index=False)
with open(metrics_json_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_summary, f, indent=2)

print("=" * 60)
print(f"📋 TỔNG KẾT KẾT QUẢ EXP 4 (DUAL-BRANCH HYBRID NET):")
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

    out_file = "d:/Development/projects/dual_branch_bone_age/kaggle_notebooks/02b_dual_branch_cross_attention_film.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1, ensure_ascii=False)
    print(f"Generated {out_file} successfully!")

if __name__ == "__main__":
    create_notebook()
