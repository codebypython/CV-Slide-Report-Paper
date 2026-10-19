"""Scientific Training Engine for Dual-Branch Bone Age Assessment.

Key technical specifications:
- Smooth L1 Loss (Huber) for initial gradient stability and outlier mitigation.
- Automatic Mixed Precision (AMP) with torch.cuda.amp.GradScaler.
- Gradient Accumulation supporting small physical batch size with larger effective batch size.
- Gradient norm clipping (1.0).
- Gradual unfreezing with discriminative learning rates across backbone and interaction heads.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional, Callable

from ..utils.logger import setup_logger

logger = setup_logger("Trainer")


def get_optimizer_param_groups(
    model: nn.Module,
    lr_backbone: float = 5e-5,
    lr_head: float = 2e-4,
    weight_decay: float = 1e-4
) -> list:
    """Configures discriminative learning rates between heavy backbones and interaction heads."""
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "convnext" in name or "swin" in name or "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    groups = []
    if backbone_params:
        groups.append({
            "params": backbone_params,
            "lr": lr_backbone,
            "weight_decay": weight_decay
        })
    if head_params:
        groups.append({
            "params": head_params,
            "lr": lr_head,
            "weight_decay": weight_decay
        })

    return groups


def set_backbone_freeze(model: nn.Module, freeze: bool = True) -> None:
    """Freezes or unfreezes backbone feature extractors."""
    for name, param in model.named_parameters():
        if "convnext" in name or "swin" in name or "backbone" in name:
            param.requires_grad = not freeze


def forward_step(model: nn.Module, batch: tuple, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniform forward pass handling both dual-branch and single-branch models."""
    img_conv, img_swin, gender, targets, meta = batch
    img_conv = img_conv.to(device, non_blocking=True)
    img_swin = img_swin.to(device, non_blocking=True)
    gender = gender.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    # Check model input requirements
    model_cls_name = model.__class__.__name__
    if model_cls_name in ["InceptionV3Baseline", "ConvNeXtSingleBranch"]:
        preds = model(img_conv, gender)
    elif model_cls_name == "SwinTSingleBranch":
        preds = model(img_swin, gender)
    else:
        # Dual-branch architectures
        preds = model(img_conv, img_swin, gender)

    return preds, targets


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    criterion: nn.Module,
    gradient_accumulation_steps: int = 1,
    max_norm: float = 1.0
) -> Tuple[float, float]:
    """Executes one training epoch with gradient accumulation and AMP."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    optimizer.zero_grad()
    use_cuda = device.type == "cuda"

    for step, batch in enumerate(dataloader):
        # Mixed precision context
        if use_cuda and scaler is not None:
            with torch.cuda.amp.autocast():
                preds, targets = forward_step(model, batch, device)
                loss = criterion(preds, targets)
                loss_scaled = loss / gradient_accumulation_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            preds, targets = forward_step(model, batch, device)
            loss = criterion(preds, targets)
            loss_scaled = loss / gradient_accumulation_steps
            loss_scaled.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()
                optimizer.zero_grad()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_mae += torch.sum(torch.abs(preds.detach() - targets)).item()
        total_samples += batch_size

    epoch_loss = total_loss / max(total_samples, 1)
    epoch_mae = total_mae / max(total_samples, 1)
    return epoch_loss, epoch_mae


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module
) -> Tuple[float, float]:
    """Evaluates validation loss and MAE."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    for batch in dataloader:
        preds, targets = forward_step(model, batch, device)
        loss = criterion(preds, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_mae += torch.sum(torch.abs(preds - targets)).item()
        total_samples += batch_size

    epoch_loss = total_loss / max(total_samples, 1)
    epoch_mae = total_mae / max(total_samples, 1)
    return epoch_loss, epoch_mae


class BoneAgeTrainingOrchestrator:
    """Manages the two-stage training regime, unfreezing, checkpointing, and scheduler stepping."""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        total_epochs: int = 50,
        warmup_epochs: int = 5,
        lr_head_warmup: float = 1e-3,
        lr_backbone_finetune: float = 5e-5,
        lr_head_finetune: float = 2e-4,
        gradient_accumulation_steps: int = 4,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_name: str = "best_model.pt"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.lr_head_warmup = lr_head_warmup
        self.lr_backbone_finetune = lr_backbone_finetune
        self.lr_head_finetune = lr_head_finetune
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.criterion = nn.SmoothL1Loss(beta=1.0)
        self.scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        self.best_mae = float("inf")
        self.history = []

    def run(self) -> Dict[str, Any]:
        """Runs the two-stage training protocol."""
        logger.info(f"Initiating Training Regime: {self.total_epochs} epochs total.")
        logger.info(f"Phase 1: Warmup ({self.warmup_epochs} epochs) with frozen backbones.")

        # Phase 1: Freeze backbones
        set_backbone_freeze(self.model, freeze=True)
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr_head_warmup,
            weight_decay=1e-4
        )

        for epoch in range(1, self.total_epochs + 1):
            # Phase transition at warmup_epochs + 1
            if epoch == (self.warmup_epochs + 1):
                logger.info("Transitioning to Phase 2: Unfreezing backbones for End-to-End Fine-Tuning.")
                set_backbone_freeze(self.model, freeze=False)
                param_groups = get_optimizer_param_groups(
                    self.model,
                    lr_backbone=self.lr_backbone_finetune,
                    lr_head=self.lr_head_finetune
                )
                optimizer = torch.optim.AdamW(param_groups)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=(self.total_epochs - self.warmup_epochs),
                    eta_min=1e-6
                )
            else:
                scheduler = None

            t0 = time.time()
            train_loss, train_mae = train_one_epoch(
                self.model, self.train_loader, optimizer, self.scaler,
                self.device, self.criterion, self.gradient_accumulation_steps
            )

            val_loss, val_mae = evaluate_epoch(
                self.model, self.val_loader, self.device, self.criterion
            )

            if scheduler is not None:
                scheduler.step()

            elapsed = time.time() - t0
            is_best = val_mae < self.best_mae
            if is_best:
                self.best_mae = val_mae
                ckpt_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_mae": val_mae,
                    "best_mae": self.best_mae
                }, ckpt_path)

            epoch_log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "is_best": is_best,
                "time_sec": elapsed
            }
            self.history.append(epoch_log)

            logger.info(
                f"Epoch [{epoch:02d}/{self.total_epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.2f} mo | "
                f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f} mo | "
                f"Best Val MAE: {self.best_mae:.2f} mo ({elapsed:.1f}s)"
            )

        return {
            "best_mae": self.best_mae,
            "history": self.history
        }
