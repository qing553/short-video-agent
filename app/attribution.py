import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr


def _is_numeric_target(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def compute_feature_importance(df: pd.DataFrame, target: str, n_top: int = 8):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    feature_df = df.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
    y = df[target]

    if feature_df.shape[1] == 0:
        return {"top_features": []}

    X = feature_df.fillna(0)
    target_values = y.fillna(0)

    if _is_numeric_target(target_values):
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)

    model.fit(X, target_values)
    perm = permutation_importance(model, X, target_values, n_repeats=10, random_state=42, n_jobs=1)

    importances = dict(zip(X.columns, perm.importances_mean))
    direction = {}
    for col in X.columns:
        try:
            corr = pearsonr(X[col], target_values)[0]
        except Exception:
            corr = 0.0
        if np.isnan(corr):
            corr = 0.0
        direction[col] = "positive" if corr > 0 else ("negative" if corr < 0 else "neutral")

    ranked = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:n_top]
    top_features = []
    for feature, importance in ranked:
        top_features.append({
            "feature": feature,
            "importance": float(importance),
            "direction": direction.get(feature, "neutral"),
        })

    return {"top_features": top_features}
