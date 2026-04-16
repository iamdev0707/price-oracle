from pydantic import BaseModel, Field
from typing import List, Optional


class PredictRequest(BaseModel):
    category: str = Field(..., example="Electronics")
    brand: str = Field(..., example="BrandA")
    days_on_market: int = Field(..., ge=1, le=3650, example=30)
    competitor_avg_price: float = Field(..., gt=0, example=999.99)
    stock_quantity: int = Field(..., ge=0, example=50)
    rating: float = Field(..., ge=1.0, le=5.0, example=4.2)
    num_reviews: int = Field(..., ge=0, example=120)
    discount_percent: float = Field(..., ge=0, le=100, example=10.0)
    is_featured: int = Field(..., ge=0, le=1, example=1)
    season_index: float = Field(..., ge=0.1, le=3.0, example=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "category": "Electronics",
                "brand": "BrandA",
                "days_on_market": 30,
                "competitor_avg_price": 999.99,
                "stock_quantity": 50,
                "rating": 4.2,
                "num_reviews": 120,
                "discount_percent": 10.0,
                "is_featured": 1,
                "season_index": 1.0,
            }
        }


class SHAPFeature(BaseModel):
    feature: str
    label: str
    value: float
    shap_value: float
    impact: str
    impact_amount: float


class Explanation(BaseModel):
    base_value: float
    features: List[SHAPFeature]
    top_driver: str
    summary: str


class PredictResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"
    explanation: Explanation
    model_version: str = "xgboost-v1"
    status: str = "success"
