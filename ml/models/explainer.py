import shap
import xgboost as xgb
import pandas as pd
import numpy as np
import json

MODEL_PATH = "ml/models/artifacts/xgboost_price_model.json"
FEATURES_PATH = "ml/models/artifacts/feature_cols.json"

_explainer = None
_feature_cols = None

FEATURE_LABELS = {
    "category_enc": "Product Category",
    "brand_enc": "Brand",
    "days_on_market": "Days on Market",
    "competitor_avg_price": "Competitor Avg Price",
    "stock_quantity": "Stock Quantity",
    "rating": "Customer Rating",
    "num_reviews": "Number of Reviews",
    "discount_percent": "Discount %",
    "is_featured": "Featured Listing",
    "season_index": "Season Demand",
}


def _load():
    global _explainer, _feature_cols
    if _explainer is None:
        model = xgb.XGBRegressor()
        model.load_model(MODEL_PATH)
        _explainer = shap.TreeExplainer(model)
        with open(FEATURES_PATH) as f:
            _feature_cols = json.load(f)


def explain(df: pd.DataFrame) -> dict:
    _load()
    shap_values = _explainer.shap_values(df)
    base_value = float(_explainer.expected_value)

    explanation = []
    for i, col in enumerate(df.columns):
        val = float(shap_values[0][i])
        explanation.append({
            "feature": col,
            "label": FEATURE_LABELS.get(col, col),
            "value": float(df.iloc[0][col]),
            "shap_value": round(val, 4),
            "impact": "increases" if val > 0 else "decreases",
            "impact_amount": round(abs(val), 2),
        })

    explanation.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    top = explanation[0]
    if top["impact"] == "increases":
        summary = (
            f"Price is mainly driven UP by {top['label']} "
            f"(+${top['impact_amount']})"
        )
    else:
        summary = (
            f"Price is mainly driven DOWN by {top['label']} "
            f"(-${top['impact_amount']})"
        )

    return {
        "base_value": round(base_value, 2),
        "features": explanation,
        "top_driver": explanation[0]["label"],
        "summary": summary,
    }
