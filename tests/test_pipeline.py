import pandas as pd

from app.cleaner import clean_df
from app.attribution import compute_feature_importance


def test_clean_df_basic():
    df = pd.DataFrame({
        "views": [100, 200, None],
        "clicks": [12, 30, 20],
        "sales": [1000, 2000, 1500],
    })
    cleaned, summary = clean_df(df)
    assert summary["n_rows"] == 3
    assert "ctr" in cleaned.columns
    assert cleaned["views"].isna().sum() == 0


def test_feature_importance_has_top_features():
    df = pd.DataFrame({
        "views": [100, 200, 300, 400, 500],
        "clicks": [10, 30, 50, 60, 80],
        "sales": [50, 100, 180, 200, 250],
    })
    result = compute_feature_importance(df, "sales", n_top=2)
    assert "top_features" in result
    assert len(result["top_features"]) > 0
