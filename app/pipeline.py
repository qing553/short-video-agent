import os
import json

import pandas as pd

from app.cleaner import clean_df
from app.attribution import compute_feature_importance


def run_pipeline(df: pd.DataFrame, target: str, llm_client, n_top: int = 6):
    df_clean, summary = clean_df(df)

    attr_result = compute_feature_importance(df_clean, target, n_top=n_top)
    top_features = attr_result.get("top_features", [])

    feature_names = [f["feature"] for f in top_features if f["feature"] in df_clean.columns]
    examples = []
    if feature_names:
        for _, row in df_clean.head(3).iterrows():
            examples.append({k: row.get(k, None) for k in feature_names})

    data_summary = {
        "n_rows": summary["n_rows"],
        "n_cols": summary["n_cols"],
        "numeric_cols": summary["numeric_cols"],
        "derived_features": summary["derived_features"],
    }

    llm_output = llm_client.explain_attribution(top_features, data_summary, examples, n_actions=5)

    report = {
        "clean_summary": summary,
        "top_features": top_features,
        "llm_explanation": llm_output,
    }
    return report
