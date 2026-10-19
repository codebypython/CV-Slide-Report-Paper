"""Stage 2: Fair Benchmarking - Replicating the RSNA 2017 Inception-v3 Baseline.

Trains Inception-v3 with gender feature concatenation on the standardized
Stratified 5-Fold split generated in Stage 1, using identical preprocessing.
Evaluates overall MAE, RMSE, R2, and puberty transition subgroup performance.
"""

import os
import sys
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.base_config import ExperimentConfig
from src.data.dataset import RSNABoneAgeDataset
from src.data.transforms import DualBranchTransform
from src.models.baselines import InceptionV3Baseline
from src.engine.trainer import BoneAgeTrainingOrchestrator
from src.engine.evaluator import ClinicalEvaluator
from src.utils.logger import setup_logger, save_json
from src.utils.reproducibility import set_seed

logger = setup_logger("Stage2_InceptionBaseline")


def run_inception_baseline(fold_idx: int = 0, epochs: int = 10, batch_size: int = 8):
    logger.info("=== GIAI ĐOẠN 2: TÁI LẬP BASELINE CHUẨN (INCEPTION-V3 2017) ===")
    config = ExperimentConfig()
    set_seed(config.seed)
    device = torch.device(config.device)
    logger.info(f"Target compute device: {device}")

    # Load Stage 1 Stratified Folds
    folds_path = os.path.join(config.data.data_dir, "stratified_5folds.csv")
    if not os.path.exists(folds_path):
        raise FileNotFoundError(f"Stratified folds not found at {folds_path}. Run Stage 1 first.")

    df_all = pd.read_csv(folds_path)
    train_df = df_all[df_all["fold"] != fold_idx].reset_index(drop=True)
    val_df = df_all[df_all["fold"] == fold_idx].reset_index(drop=True)
    logger.info(f"Fold {fold_idx}: Train set = {len(train_df)}, Val set = {len(val_df)}")

    # Dataloaders with identical letterboxing & CLAHE preprocessing
    train_tf = DualBranchTransform(is_training=True)
    val_tf = DualBranchTransform(is_training=False)

    train_ds = RSNABoneAgeDataset(train_df, config.data.images_dir, transform=train_tf, is_training=True)
    val_ds = RSNABoneAgeDataset(val_df, config.data.images_dir, transform=val_tf, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize Inception-v3 Baseline Model
    logger.info("Building Inception-v3 Baseline architecture with Gender concatenation...")
    model = InceptionV3Baseline(
        model_name=config.model.inception_model_name,
        dropout_rate=config.model.dropout_rate,
        pretrained=config.model.pretrained
    ).to(device)

    # Training Orchestration
    orchestrator = BoneAgeTrainingOrchestrator(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        total_epochs=epochs,
        warmup_epochs=min(2, epochs // 3),
        lr_head_warmup=1e-3,
        lr_backbone_finetune=5e-5,
        lr_head_finetune=2e-4,
        gradient_accumulation_steps=2,
        checkpoint_dir=config.training.checkpoint_dir,
        checkpoint_name="inception_baseline_best.pt"
    )

    train_summary = orchestrator.run()

    # Clinical Statistical Evaluation with 1000x Bootstrap
    logger.info("Running rigorous Clinical Statistical Evaluation with 1000x Bootstrap CI...")
    evaluator = ClinicalEvaluator(n_bootstrap=1000, seed=config.seed)
    df_preds = evaluator.collect_predictions(model, val_loader, device)
    eval_metrics = evaluator.evaluate_dataframe(df_preds)

    report_md = evaluator.format_markdown_report(eval_metrics, model_name="Exp 0: Inception-v3 (RSNA 2017 Baseline)")
    logger.info("\n" + report_md)

    # Save outputs
    out_dir = os.path.join(config.training.output_dir, "exp0_baseline")
    os.makedirs(out_dir, exist_ok=True)
    df_preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    save_json(eval_metrics, os.path.join(out_dir, "metrics.json"))
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info("=== GIAI ĐOẠN 2 HOÀN TẤT THÀNH CÔNG ===")
    return eval_metrics


if __name__ == "__main__":
    run_inception_baseline(fold_idx=0, epochs=4, batch_size=8)
