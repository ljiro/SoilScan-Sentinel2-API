from fastapi import FastAPI

from app.api.predict import router as predict_router

app = FastAPI(
    title="SoilScan Sentinel-2 API",
    description=(
        "Accepts a GIS polygon, queries locally downloaded Sentinel-2 band data "
        "and SoilGrids soil priors, and returns soil nutrient predictions "
        "(N, P, K, pH) using trained Random Forest / SVM classifiers."
    ),
    version="1.0.0",
)

app.include_router(predict_router, tags=["Inference"])


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
