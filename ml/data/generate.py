import pandas as pd
import numpy as np
import os

def generate_pricing_data(n_samples: int = 1000, save: bool = True) -> pd.DataFrame:
    np.random.seed(42)

    categories = ["Electronics", "Fashion", "Home", "Sports", "Books"]
    brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]

    data = {
        "category": np.random.choice(categories, n_samples),
        "brand": np.random.choice(brands, n_samples),
        "days_on_market": np.random.randint(1, 365, n_samples),
        "competitor_avg_price": np.random.uniform(50, 2000, n_samples),
        "stock_quantity": np.random.randint(1, 500, n_samples),
        "rating": np.round(np.random.uniform(1.0, 5.0, n_samples), 1),
        "num_reviews": np.random.randint(0, 5000, n_samples),
        "discount_percent": np.random.uniform(0, 50, n_samples),
        "is_featured": np.random.randint(0, 2, n_samples),
        "season_index": np.random.uniform(0.5, 1.5, n_samples),
    }

    df = pd.DataFrame(data)

    # Encode categoricals for price formula
    cat_map = {c: i * 100 for i, c in enumerate(categories)}
    brand_map = {b: i * 50 for i, b in enumerate(brands)}

    df["price"] = (
        df["competitor_avg_price"] * 0.95
        + df["category"].map(cat_map)
        + df["brand"].map(brand_map)
        + df["rating"] * 20
        + df["num_reviews"] * 0.01
        - df["days_on_market"] * 0.1
        - df["discount_percent"] * 2
        + df["is_featured"] * 50
        + df["season_index"] * 30
        + np.random.normal(0, 20, n_samples)
    ).clip(10, 5000).round(2)

    if save:
        os.makedirs("ml/data", exist_ok=True)
        df.to_csv("ml/data/pricing_data.csv", index=False)
        print(f"[+] Generated {n_samples} samples -> ml/data/pricing_data.csv")

    return df


if __name__ == "__main__":
    df = generate_pricing_data()
    print(df.head())
    print(f"\nPrice stats:\n{df['price'].describe()}")
