"""Stage 3: Proposed Architecture Training (ConvNeXt + SwinT + Cross-Attention + FiLM).

Executes end-to-end training of the proposed dual-branch model:
- ConvNeXt for micro-morphology.
- Swin Transformer for macro-structural reasoning.
- Cross-Attention module for multi-scale feature dialogue.
- FiLM layer for clinical gender modulation.
- Two-stage training regime with gradual unfreezing and discriminative learning rates.
"""

import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.base_config import ExperimentConfig
from src.data.dataset import RSNABoneAgeDataset
from src.data.transforms import DualBranchTransform
from src.models.hybrid_bone_age import DualBranchBoneAgeNet
from src.engine.trainer import BoneAgeTrainingOrchestrator
from src.engine.evaluator import ClinicalEvaluator
from src.utils.logger import setup_logger, save_json
from src.utils.reproducibility import set_seed

logger = setup_logger("Stage3_ProposedModel")


def run_proposed_training(fold_idx: int = 0, epochs: int = 10, batch_size: int = 8):
    logger.info("=== GIAI ĐOẠN 3: HUẤN LUYỆN MÔ HÌNH ĐỀ XUẤT (DUAL-BRANCH CONVNEXT + SWIN + FILM) ===")
    config = ExperimentConfig()
    set_seed(config.seed)
    device = torch.device(config.device)
    logger.info(f"Compute device: {device}")

    # Load Stage 1 Stratified Folds
    folds_path = os.path.join(config.data.data_dir, "stratified_5folds.csv")
    if not os.path.exists(folds_path):
        raise FileNotFoundError(f"Stratified folds not found at {folds_path}. Run Stage 1 first.")

    df_all = pd.read_csv(folds_path)
    train_df = df_all[df_all["fold"] != fold_idx].reset_index(drop=True)
    val_df = df_all[df_all["fold"] == fold_idx].reset_index(drop=True)
    logger.info(f"Fold {fold_idx}: Train set = {len(train_df)}, Val set = {len(val_df)}")

    train_tf = DualBranchTransform(is_training=True)
    val_tf = DualBranchTransform(is_training=False)

    train_ds = RSNABoneAgeDataset(train_df, config.data.images_dir, transform=train_tf, is_training=True)
    val_ds = RSNABoneAgeDataset(val_df, config.data.images_dir, transform=val_tf, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Instantiate Proposed Model
    logger.info("Constructing Proposed DualBranchBoneAgeNet with Cross-Attention & FiLM...")
    model = DualBranchBoneAgeNet(
        convnext_name=config.model.convnext_model_name,
        swin_name=config.model.swin_model_name,
        embed_dim=config.model.embed_dim,
        num_heads=config.model.num_heads,
        dropout_rate=config.model.dropout_rate,
        use_film=True,
        use_cross_attention=True,
        bidirectional_attention=False,
        pretrained=config.model.pretrained
    ).to(device)

    # Two-Stage Orchestration
    orchestrator = BoneAgeTrainingOrchestrator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        total_epochs=epochs,
        warmup_epochs=max(1, epochs // 2),
        lr_head_warmup=config.training.lr_head_warmup,
        lr_backbone_finetune=config.training.lr_backbone_finetune,
        lr_head_finetune=config.training.lr_head_finetune,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        checkpoint_dir=config.training.checkpoint_dir,
        checkpoint_name="proposed_model_best.pt"
    )

    train_summary = orchestrator.run()

    # Clinical Statistical Evaluation with 1000x Bootstrap
    logger.info("Executing Clinical Evaluation with 1000x Bootstrap CI...")
    evaluator = ClinicalEvaluator(n_bootstrap=1000, seed=config.seed)
    df_preds = evaluator.collect_predictions(model, val_loader, device)
    eval_metrics = evaluator.evaluate_dataframe(df_preds)

    report_md = evaluator.format_markdown_report(
        eval_metrics,
        model_name="Exp 5 (Proposed): Dual-Branch ConvNeXt + SwinT + Cross-Attention + FiLM"
    )
    logger.info("\n" + report_md)

    # Save outputs
    out_dir = os.path.join(config.training.output_dir, "exp5_proposed")
    os.makedirs(out_dir, exist_ok=True)
    df_preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    save_json(eval_metrics, os.path.join(out_dir, "metrics.json"))
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info("=== GIAI ĐOẠN 3 HOÀN TẤT THÀNH CÔNG ===")
    return eval_metrics


if __name__ == "__main__":
    run_proposed_training(fold_idx=0, epochs=2, batch_size=8)
