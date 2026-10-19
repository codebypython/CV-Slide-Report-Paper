"""Clinical evaluation engine computing statistical metrics, 95% Bootstrap CI, and subgroup diagnostics."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, List, Tuple, Optional

from .trainer import forward_step


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Computes point estimate and 95% Confidence Interval via non-parametric bootstrap resampling."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point_est = float(metric_fn(y_true, y_pred))

    if n < 5:
        return point_est, point_est, point_est

    boot_scores = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        boot_score = metric_fn(y_true[indices], y_pred[indices])
        boot_scores.append(boot_score)

    lower_ci = float(np.percentile(boot_scores, 100 * (alpha / 2)))
    upper_ci = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return point_est, lower_ci, upper_ci


def is_pubertal_acceleration_cohort(bone_age_months: float, is_male: bool) -> bool:
    """Clinical criterion for rapid pubertal growth spurt:
    
    Female: 11 - 14 years (132 to 168 months)
    Male: 12 - 15 years (144 to 180 months)
    """
    if is_male:
        return 144.0 <= bone_age_months <= 180.0
    else:
        return 132.0 <= bone_age_months <= 168.0


class ClinicalEvaluator:
    """Evaluates bone age regression models across overall metrics and clinical developmental subgroups."""
    def __init__(self, n_bootstrap: int = 1000, seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    @torch.no_grad()
    def collect_predictions(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ) -> pd.DataFrame:
        """Runs inference over dataloader and aggregates predictions with clinical patient metadata."""
        model.eval()
        records = []

        for batch in dataloader:
            preds, targets = forward_step(model, batch, device)
            _, _, _, _, meta = batch

            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for i in range(len(preds_np)):
                p = float(preds_np[i])
                t = float(targets_np[i])

                # Robust extraction across tensors, lists, and scalars
                if torch.is_tensor(meta["id"]):
                    sample_id = meta["id"][i].item()
                elif isinstance(meta["id"], (list, tuple)):
                    sample_id = meta["id"][i]
                else:
                    sample_id = meta["id"]

                if torch.is_tensor(meta["male"]):
                    is_male = bool(meta["male"][i].item())
                elif isinstance(meta["male"], (list, tuple)):
                    is_male = bool(meta["male"][i])
                else:
                    is_male = bool(meta["male"])

                if isinstance(meta["age_bracket"], (list, tuple)):
                    bracket = str(meta["age_bracket"][i])
                elif torch.is_tensor(meta["age_bracket"]):
                    bracket = str(meta["age_bracket"][i].item())
                else:
                    bracket = str(meta["age_bracket"])

                is_pubertal = is_pubertal_acceleration_cohort(t, is_male)

                records.append({
                    "id": sample_id,
                    "target_age": t,
                    "pred_age": p,
                    "abs_error": abs(p - t),
                    "sq_error": (p - t) ** 2,
                    "is_male": is_male,
                    "age_bracket": bracket,
                    "is_pubertal_spurt": is_pubertal
                })

        return pd.DataFrame(records)

    def evaluate_dataframe(self, df_preds: pd.DataFrame) -> Dict[str, Any]:
        """Calculates point estimates and 95% Bootstrap CIs for overall and subgroup metrics."""
        y_true = df_preds["target_age"].values
        y_pred = df_preds["pred_age"].values

        # Overall metrics
        mae, mae_low, mae_high = compute_bootstrap_ci(
            y_true, y_pred, mean_absolute_error, self.n_bootstrap, seed=self.seed
        )
        rmse_fn = lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))
        rmse, rmse_low, rmse_high = compute_bootstrap_ci(
            y_true, y_pred, rmse_fn, self.n_bootstrap, seed=self.seed
        )
        r2, r2_low, r2_high = compute_bootstrap_ci(
            y_true, y_pred, r2_score, self.n_bootstrap, seed=self.seed
        )

        # Sex subgroups
        male_mask = df_preds["is_male"] == True
        female_mask = ~male_mask

        male_mae = float(mean_absolute_error(y_true[male_mask], y_pred[male_mask])) if male_mask.sum() > 0 else 0.0
        female_mae = float(mean_absolute_error(y_true[female_mask], y_pred[female_mask])) if female_mask.sum() > 0 else 0.0

        # Pubertal acceleration window subgroup (Critical research focus)
        puberty_mask = df_preds["is_pubertal_spurt"] == True
        puberty_mae, pub_low, pub_high = compute_bootstrap_ci(
            y_true[puberty_mask], y_pred[puberty_mask], mean_absolute_error, self.n_bootstrap, seed=self.seed
        ) if puberty_mask.sum() > 0 else (0.0, 0.0, 0.0)

        # Age bracket breakdown
        bracket_results = {}
        for bracket, grp in df_preds.groupby("age_bracket"):
            b_true = grp["target_age"].values
            b_pred = grp["pred_age"].values
            b_mae = float(mean_absolute_error(b_true, b_pred))
            bracket_results[str(bracket)] = {
                "count": len(grp),
                "mae": b_mae
            }

        return {
            "overall": {
                "mae": mae,
                "mae_95ci": (mae_low, mae_high),
                "rmse": rmse,
                "rmse_95ci": (rmse_low, rmse_high),
                "r2": r2,
                "r2_95ci": (r2_low, r2_high),
                "total_samples": len(df_preds)
            },
            "sex_breakdown": {
                "male_mae": male_mae,
                "male_count": int(male_mask.sum()),
                "female_mae": female_mae,
                "female_count": int(female_mask.sum())
            },
            "pubertal_spurt_cohort": {
                "pubertal_mae": puberty_mae,
                "pubertal_95ci": (pub_low, pub_high),
                "pubertal_count": int(puberty_mask.sum())
            },
            "age_bracket_breakdown": bracket_results
        }

    def format_markdown_report(self, results: Dict[str, Any], model_name: str = "Proposed Model") -> str:
        """Formats evaluation metrics into an academic markdown table."""
        ov = results["overall"]
        sb = results["sex_breakdown"]
        pb = results["pubertal_spurt_cohort"]

        report = f"""### Clinical Evaluation Report: {model_name}

| Metric | Point Estimate | 95% Confidence Interval | Clinical Interpretation |
| :--- | :--- | :--- | :--- |
| **MAE (Overall)** | **{ov['mae']:.2f} months** | [{ov['mae_95ci'][0]:.2f}, {ov['mae_95ci'][1]:.2f}] | Primary scientific benchmark |
| **RMSE** | {ov['rmse']:.2f} months | [{ov['rmse_95ci'][0]:.2f}, {ov['rmse_95ci'][1]:.2f}] | Sensitivity to large outliers |
| **$R^2$ Score** | {ov['r2']:.4f} | [{ov['r2_95ci'][0]:.4f}, {ov['r2_95ci'][1]:.4f}] | Proportion of developmental variance explained |
| **Male MAE** | {sb['male_mae']:.2f} months | N={sb['male_count']} | Androgen-driven ossification trajectory |
| **Female MAE** | {sb['female_mae']:.2f} months | N={sb['female_count']} | Estrogen-accelerated growth plate fusion |
| **Pubertal Spurt MAE** | **{pb['pubertal_mae']:.2f} months** | [{pb['pubertal_95ci'][0]:.2f}, {pb['pubertal_95ci'][1]:.2f}] | **F: 11-14y / M: 12-15y critical transition cohort** |

#### Age-Bracket Error Distribution:
"""
        for bracket, b_data in results["age_bracket_breakdown"].items():
            report += f"- **{bracket}** (N={b_data['count']}): MAE = {b_data['mae']:.2f} months\n"

        return report
