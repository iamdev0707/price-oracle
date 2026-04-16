import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import json

FEATURE_COLS = [
    "category_enc", "brand_enc", "days_on_market",
    "competitor_avg_price", "stock_quantity", "rating",
    "num_reviews", "discount_percent", "is_featured", "season_index"
]
TARGET_COL = "price"
MODEL_DIR = "ml/models/artifacts"
MODEL_PATH = f"{MODEL_DIR}/xgboost_price_model.json"
ENCODER_PATH = f"{MODEL_DIR}/label_encoders.pkl"
FEATURES_PATH = f"{MODEL_DIR}/feature_cols.json"


def load_and_prepare(csv_path: str = "ml/data/pricing_data.csv"):
    df = pd.read_csv(csv_path)

    encoders = {}
    for col in ["category", "brand"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y, encoders


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("[1/4] Loading data...")
    X, y, encoders = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    print("[2/4] Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    print("[3/4] Evaluating...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"      MAE: ${mae:.2f} | R2: {r2:.4f}")

    print("[4/4] Saving model + encoders...")
    model.save_model(MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(FEATURE_COLS, f)

    print(f"\n[+] Model saved to {MODEL_PATH}")
    print(f"[+] Encoders saved to {ENCODER_PATH}")
    return model, encoders


if __name__ == "__main__":
    train()
