"""Configuration module for Dual-Branch Bone Age Assessment Research Project.

Defines all hyperparameters, data protocol specifications, ablation flags,
and training parameters in a strictly typed, reproducible dataclass.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os


@dataclass
class ModelConfig:
    # Backbone architectures
    convnext_model_name: str = "convnext_small"
    swin_model_name: str = "swin_small_patch4_window7_224"
    inception_model_name: str = "inception_v3"
    
    # Feature dimensions
    embed_dim: int = 512
    num_heads: int = 8
    attention_dropout: float = 0.1
    film_hidden_dim: int = 128
    dropout_rate: float = 0.3
    
    # Pretrained weights flag
    pretrained: bool = False  # Set True when downloading ImageNet weights online
    
    # Model type for ablation study
    # Choices: 'inception_baseline', 'convnext_single', 'swin_single',
    #          'naive_concat', 'cross_attention_only', 'proposed_film_cross_attn'
    architecture_type: str = "proposed_film_cross_attn"


@dataclass
class DataConfig:
    # Image resolution for dual branches
    conv_img_size: int = 384     # High resolution for micro-morphology
    swin_img_size: int = 224     # Standard window size for macro-reasoning
    raw_img_size: int = 512      # Square letterboxed canonical size
    
    # Normalization parameters (ImageNet default)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    
    # Age stratification boundaries (in months)
    # <24 mo (infancy), 24-120 mo (childhood), 121-180 mo (puberty), >180 mo (post-puberty)
    age_bins: List[int] = field(default_factory=lambda: [0, 24, 120, 180, 250])
    age_bin_labels: List[str] = field(
        default_factory=lambda: ["Infancy (<2y)", "Early Childhood (2-10y)", "Puberty (10-15y)", "Post-Puberty (>15y)"]
    )
    
    num_folds: int = 5
    random_seed: int = 42
    
    # Dataset paths
    data_dir: str = "d:/Development/projects/dual_branch_bone_age/data"
    csv_file: str = "d:/Development/projects/dual_branch_bone_age/data/boneage_training_dataset.csv"
    images_dir: str = "d:/Development/projects/dual_branch_bone_age/data/images"


@dataclass
class TrainingConfig:
    # Training regime
    epochs_stage1_warmup: int = 5    # Frozen backbones, only train heads
    epochs_stage2_finetune: int = 45 # End-to-end full fine-tuning
    total_epochs: int = 50
    
    # Learning rates
    lr_head_warmup: float = 1e-3
    lr_backbone_finetune: float = 5e-5
    lr_head_finetune: float = 2e-4
    weight_decay: float = 1e-4
    
    # Batch scheduling & Hardware
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 8 * 4 = 32
    num_workers: int = 0                  # Safe for Windows multiprocessing
    use_amp: bool = True
    gradient_clip_norm: float = 1.0
    
    # Scheduler
    min_lr: float = 1e-6
    warmup_epochs: int = 3
    
    # Checkpointing
    checkpoint_dir: str = "d:/Development/projects/dual_branch_bone_age/checkpoints"
    output_dir: str = "d:/Development/projects/dual_branch_bone_age/outputs"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    seed: int = 42


def get_ablation_configs() -> dict:
    """Returns configuration dictionary for Exp 0 to Exp 5."""
    return {
        "exp0_inception_baseline": {
            "name": "Exp 0 (Baseline): Inception-v3 + Gender Concat",
            "arch": "inception_baseline",
            "desc": "Fair RSNA 2017 winning baseline replication on unified pipeline"
        },
        "exp1_convnext_single": {
            "name": "Exp 1: ConvNeXt-Small + Gender Concat",
            "arch": "convnext_single",
            "desc": "Modern CNN inductive bias isolating micro-morphological feature learning"
        },
        "exp2_swint_single": {
            "name": "Exp 2: Swin-Small + Gender Concat",
            "arch": "swint_single",
            "desc": "Hierarchical Vision Transformer isolating macro-structural relation learning"
        },
        "exp3_naive_concat": {
            "name": "Exp 3: ConvNeXt + SwinT (Naive Concat/Add)",
            "arch": "naive_concat",
            "desc": "Dual-branch combination without cross-modal interaction"
        },
        "exp4_cross_attention": {
            "name": "Exp 4: ConvNeXt + SwinT + Cross-Attention",
            "arch": "cross_attention_only",
            "desc": "Active cross-attention interaction between micro and macro features"
        },
        "exp5_proposed_film": {
            "name": "Exp 5 (Proposed): ConvNeXt + SwinT + Cross-Attention + FiLM",
            "arch": "proposed_film_cross_attn",
            "desc": "Full proposed architecture with clinical gender conditioning via FiLM"
        }
    }
