from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers.predict import router as predict_router

app = FastAPI(
    title="PriceOracle",
    description="AI-Powered Autonomous Pricing Intelligence System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/", tags=["Root"])
def root():
    return {
        "project": "PriceOracle",
        "status": "running",
        "docs": "/docs",
        "predict": "/api/v1/predict",
    }
