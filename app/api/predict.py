from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.predict import PredictRequest, PredictResponse
from app.services.polygon_sampler import sample_polygon
from app.services.sentinel2_extractor import extract_bands
from app.services.soilgrids_extractor import extract_soilgrids
from app.services.terrain_extractor import extract_terrain
from app.services.spectral_indices import add_spectral_indices
from app.services.predictor import load_models, predict_all

router = APIRouter()
_models = None


def _get_models():
    global _models
    if _models is None:
        _models = load_models(settings.models_dir)
    return _models


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    warnings: list[str] = []

    # --- 1. Sample grid points inside polygon ---
    try:
        points = sample_polygon(
            req.polygon,
            spacing_m=req.sample_spacing_m,
            max_points=settings.max_sample_points,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid polygon: {exc}")

    if not points:
        raise HTTPException(status_code=422, detail="Polygon produced no sample points.")

    if len(points) == settings.max_sample_points:
        warnings.append(
            f"Sample count capped at {settings.max_sample_points}. "
            "Increase sample_spacing_m or the SOILSCAN_MAX_SAMPLE_POINTS env var for full coverage."
        )

    lons = [p[0] for p in points]
    lats = [p[1] for p in points]

    # --- 2. Extract Sentinel-2 bands ---
    band_means, band_stds = extract_bands(settings.sentinel2_dir, points)
    if band_means is None:
        raise HTTPException(
            status_code=503,
            detail="No Sentinel-2 tiles found. Ensure .SAFE directories exist in data/sentinel2/.",
        )

    # --- 3. Extract SoilGrids priors ---
    soilgrids_data = extract_soilgrids(settings.soilgrids_dir, points)

    # --- 4. Extract terrain features ---
    terrain_data = extract_terrain(settings.dem_path, points)
    if terrain_data is None:
        warnings.append("DEM not found — terrain features will be NaN (imputed by model).")

    # --- 5. Resolve microclimate ---
    temperature_c = req.temperature_c if req.temperature_c is not None else settings.default_temperature_c
    humidity_percent = req.humidity_percent if req.humidity_percent is not None else settings.default_humidity_percent

    # --- 6. Compute polygon area ---
    from app.services.polygon_sampler import polygon_area_ha
    area_ha = polygon_area_ha(req.polygon)

    # --- 7. Assemble features and run inference ---
    models = _get_models()
    result = predict_all(
        models=models,
        points=points,
        band_means=band_means,
        band_stds=band_stds,
        soilgrids=soilgrids_data,
        terrain=terrain_data,
        temperature_c=temperature_c,
        humidity_percent=humidity_percent,
        crop_type=req.crop_type,
    )

    return PredictResponse(
        nitrogen=result["nitrogen"],
        phosphorus=result["phosphorus"],
        potassium=result["potassium"],
        ph=result["ph"],
        sample_count=len(points),
        polygon_area_ha=round(area_ha, 4),
        warnings=warnings,
    )
