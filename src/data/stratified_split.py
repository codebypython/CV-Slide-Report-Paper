"""Stratified 5-Fold Cross-Validation protocol for clinical bone age datasets.

Scientific justification:
Standard random splits suffer from distribution skewness across rapid developmental
windows (infancy vs. puberty) and sex-dimorphic ossification velocity.
This module creates a joint stratification key:
    strat_key = f"{age_bin}_{male_status}"
ensuring balanced representation of all pediatric developmental stages across every fold.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple, Optional
import os


def assign_age_bracket(bone_age_months: float, age_bins: Optional[List[int]] = None) -> str:
    """Categorizes pediatric bone age in months into clinical developmental brackets.
    
    Bins:
    - Bin 0: < 24 months (< 2 years, Infancy / Toddlerhood)
    - Bin 1: 24 - 120 months (2 - 10 years, Early-to-Middle Childhood)
    - Bin 2: 121 - 180 months (10 - 15 years, Pubertal Growth Acceleration)
    - Bin 3: > 180 months (> 15 years, Late Adolescence / Skeletal Maturity)
    """
    if age_bins is None:
        age_bins = [0, 24, 120, 180, 250]

    if bone_age_months < age_bins[1]:
        return "Infancy (<2y)"
    elif bone_age_months <= age_bins[2]:
        return "Childhood (2-10y)"
    elif bone_age_months <= age_bins[3]:
        return "Puberty (10-15y)"
    else:
        return "Post-Puberty (>15y)"


def generate_stratified_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    age_col: str = "boneage",
    gender_col: str = "male",
    output_col: str = "fold"
) -> pd.DataFrame:
    """Assigns cross-validation fold indices stratified jointly by age bracket and biological sex."""
    df_copy = df.copy()

    # Create age bracket feature
    df_copy["age_bracket"] = df_copy[age_col].apply(assign_age_bracket)

    # Convert gender to boolean / int representation
    if df_copy[gender_col].dtype == object or df_copy[gender_col].dtype == bool:
        male_int = df_copy[gender_col].apply(lambda x: 1 if str(x).strip().lower() in ["1", "true", "male", "m"] else 0)
    else:
        male_int = df_copy[gender_col].astype(int)

    df_copy["male_int"] = male_int

    # Joint stratification label
    df_copy["strat_label"] = df_copy["age_bracket"].astype(str) + "_" + df_copy["male_int"].astype(str)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df_copy[output_col] = -1

    for fold_idx, (_, val_idx) in enumerate(skf.split(df_copy, df_copy["strat_label"])):
        df_copy.loc[df_copy.index[val_idx], output_col] = fold_idx

    return df_copy


def verify_fold_distribution(df_folds: pd.DataFrame, fold_col: str = "fold") -> pd.DataFrame:
    """Verifies that each fold contains proportional representation of each subgroup."""
    summary = df_folds.groupby([fold_col, "age_bracket", "male_int"]).size().unstack(level=[1, 2], fill_value=0)
    return summary
