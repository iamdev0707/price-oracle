import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import json
import os

MODEL_PATH = "ml/models/artifacts/xgboost_price_model.json"
ENCODER_PATH = "ml/models/artifacts/label_encoders.pkl"
FEATURES_PATH = "ml/models/artifacts/feature_cols.json"

_model = None
_encoders = None
_feature_cols = None


def _load():
    global _model, _encoders, _feature_cols
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run: python ml/models/trainer.py"
            )
        _model = xgb.XGBRegressor()
        _model.load_model(MODEL_PATH)
        _encoders = joblib.load(ENCODER_PATH)
        with open(FEATURES_PATH) as f:
            _feature_cols = json.load(f)


def predict(
    category: str,
    brand: str,
    days_on_market: int,
    competitor_avg_price: float,
    stock_quantity: int,
    rating: float,
    num_reviews: int,
    discount_percent: float,
    is_featured: int,
    season_index: float,
) -> dict:
    _load()

    # Encode categoricals safely
    def safe_encode(encoder, value: str) -> int:
        classes = list(encoder.classes_)
        if value in classes:
            return int(encoder.transform([value])[0])
        return 0  # fallback to first class

    row = {
        "category_enc": safe_encode(_encoders["category"], category),
        "brand_enc": safe_encode(_encoders["brand"], brand),
        "days_on_market": days_on_market,
        "competitor_avg_price": competitor_avg_price,
        "stock_quantity": stock_quantity,
        "rating": rating,
        "num_reviews": num_reviews,
        "discount_percent": discount_percent,
        "is_featured": is_featured,
        "season_index": season_index,
    }

    X = pd.DataFrame([row])[_feature_cols]
    price = float(_model.predict(X)[0])

    return {
        "predicted_price": round(price, 2),
        "input_features": row,
        "df": X,
    }
