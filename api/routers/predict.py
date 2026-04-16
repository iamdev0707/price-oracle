from fastapi import APIRouter, HTTPException
from api.schemas.predict import PredictRequest, PredictResponse
from ml.models.predictor import predict
from ml.models.explainer import explain

router = APIRouter(prefix="/api/v1", tags=["Prediction"])


@router.post("/predict", response_model=PredictResponse, summary="Predict price with SHAP explanation")
def predict_price(body: PredictRequest):
    try:
        result = predict(
            category=body.category,
            brand=body.brand,
            days_on_market=body.days_on_market,
            competitor_avg_price=body.competitor_avg_price,
            stock_quantity=body.stock_quantity,
            rating=body.rating,
            num_reviews=body.num_reviews,
            discount_percent=body.discount_percent,
            is_featured=body.is_featured,
            season_index=body.season_index,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    try:
        explanation = explain(result["df"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

    return PredictResponse(
        predicted_price=result["predicted_price"],
        explanation=explanation,
    )


@router.get("/health", summary="Health check")
def health():
    return {"status": "PriceOracle running", "version": "1.0.0"}