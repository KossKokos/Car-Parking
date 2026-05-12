from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def assign_splits(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    group_col: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Assign train/val/test splits.

    Uses grouped splitting by label so the same plate string does not appear
    in multiple splits. This helps avoid leakage when duplicate labels exist.
    """
    total = train_size + val_size + test_size

    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0.")

    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col!r}")

    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    unique_groups = df[group_col].nunique()

    if unique_groups < 3:
        raise ValueError(
            "Need at least 3 unique labels to create train/val/test splits."
        )

    df = df.copy().reset_index(drop=True)
    df["split"] = None

    groups = df[group_col]

    first_split = GroupShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=random_state,
    )

    train_idx, temp_idx = next(first_split.split(df, groups=groups))

    df.loc[train_idx, "split"] = "train"

    temp_df = df.iloc[temp_idx].copy()

    val_fraction_of_temp = val_size / (val_size + test_size)

    second_split = GroupShuffleSplit(
        n_splits=1,
        train_size=val_fraction_of_temp,
        random_state=random_state,
    )

    val_rel_idx, test_rel_idx = next(
        second_split.split(temp_df, groups=temp_df[group_col])
    )

    val_idx = temp_df.iloc[val_rel_idx].index
    test_idx = temp_df.iloc[test_rel_idx].index

    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    if df["split"].isna().any():
        raise RuntimeError("Some rows were not assigned a split.")

    return df