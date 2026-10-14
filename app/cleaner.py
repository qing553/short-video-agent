import pandas as pd
import numpy as np


def clean_df(df: pd.DataFrame, numeric_fill_strategy: str = "median"):
    df = df.copy()
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if numeric_fill_strategy == "median":
            fill_value = df[col].median()
        elif numeric_fill_strategy == "mean":
            fill_value = df[col].mean()
        else:
            fill_value = 0
        df[col] = df[col].fillna(fill_value)

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        df[col] = df[col].fillna("UNKNOWN")

    if "views" in df.columns and "clicks" in df.columns:
        df["ctr"] = df["clicks"] / df["views"].replace(0, pd.NA)
        df["ctr"] = df["ctr"].fillna(0)

    if "visits" in df.columns and "orders" in df.columns:
        df["conversion_rate"] = df["orders"] / df["visits"].replace(0, pd.NA)
        df["conversion_rate"] = df["conversion_rate"].fillna(0)

    summary = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "numeric_cols": numeric_cols,
        "derived_features": [c for c in ["ctr", "conversion_rate"] if c in df.columns],
    }
    return df, summary
