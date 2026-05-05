from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.schemas.predict import PredictRequest, PredictResponse
from app.services.polygon_sampler import polygon_area_ha, sample_polygon
from app.services.sentinel2_extractor import extract_bands
from app.services.soilgrids_extractor import extract_soilgrids
from app.services.terrain_extractor import extract_terrain
from app.services.predictor import load_models, predict_all

router = APIRouter()
_models = None


def _get_models():
    global _models
    if _models is None:
        _models = load_models(settings.models_dir)
    return _models


def _bbox_to_polygon(minlon: float, minlat: float, maxlon: float, maxlat: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[
            [minlon, minlat],
            [maxlon, minlat],
            [maxlon, maxlat],
            [minlon, maxlat],
            [minlon, minlat],
        ]],
    }


async def _run_predict(
    polygon: dict,
    crop_type: str,
    temperature_c: Optional[float],
    humidity_percent: Optional[float],
    sample_spacing_m: float,
) -> PredictResponse:
    warnings: list[str] = []

    try:
        points = sample_polygon(
            polygon,
            spacing_m=sample_spacing_m,
            max_points=settings.max_sample_points,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid polygon: {exc}")

    if not points:
        raise HTTPException(status_code=422, detail="Polygon produced no sample points.")

    if len(points) == settings.max_sample_points:
        warnings.append(
            f"Sample count capped at {settings.max_sample_points}. "
            "Increase sample_spacing_m or SOILSCAN_MAX_SAMPLE_POINTS for full coverage."
        )

    band_means, band_stds = extract_bands(settings.sentinel2_dir, points)
    if band_means is None:
        raise HTTPException(
            status_code=503,
            detail="No Sentinel-2 tiles found. Ensure .SAFE directories exist in data/sentinel2/.",
        )

    soilgrids_data = extract_soilgrids(settings.soilgrids_dir, points)

    terrain_data = extract_terrain(settings.dem_path, points)
    if terrain_data is None:
        warnings.append("DEM unavailable and all fallbacks failed — terrain features are NaN.")
    elif np.all(np.isnan(terrain_data[:, 1:])):
        warnings.append("DEM not available locally — elevation from Open-Elevation API; slope/aspect/TWI are NaN.")

    temp = temperature_c if temperature_c is not None else settings.default_temperature_c
    humid = humidity_percent if humidity_percent is not None else settings.default_humidity_percent
    area_ha = polygon_area_ha(polygon)

    result = predict_all(
        models=_get_models(),
        points=points,
        band_means=band_means,
        band_stds=band_stds,
        soilgrids=soilgrids_data,
        terrain=terrain_data,
        temperature_c=temp,
        humidity_percent=humid,
        crop_type=crop_type,
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


@router.post("/predict", response_model=PredictResponse)
async def predict_post(req: PredictRequest):
    """Run soil prediction for a GeoJSON polygon."""
    return await _run_predict(
        polygon=req.polygon,
        crop_type=req.crop_type,
        temperature_c=req.temperature_c,
        humidity_percent=req.humidity_percent,
        sample_spacing_m=req.sample_spacing_m,
    )


@router.get("/predict", response_model=PredictResponse)
async def predict_get(
    minlon: float = Query(..., description="West boundary longitude", example=120.50),
    minlat: float = Query(..., description="South boundary latitude", example=16.40),
    maxlon: float = Query(..., description="East boundary longitude", example=120.51),
    maxlat: float = Query(..., description="North boundary latitude", example=16.41),
    crop_type: str = Query(default="unknown", description="Crop type (cabbage, tomato, potato, …)"),
    temperature_c: Optional[float] = Query(default=None, description="Air temperature in °C"),
    humidity_percent: Optional[float] = Query(default=None, description="Relative humidity %"),
    sample_spacing_m: float = Query(default=10.0, ge=5.0, le=100.0, description="Grid spacing in metres"),
):
    """Run soil prediction for a bounding box (minlon, minlat, maxlon, maxlat)."""
    if minlon >= maxlon or minlat >= maxlat:
        raise HTTPException(status_code=422, detail="bbox is invalid: minlon/minlat must be less than maxlon/maxlat.")

    polygon = _bbox_to_polygon(minlon, minlat, maxlon, maxlat)
    return await _run_predict(
        polygon=polygon,
        crop_type=crop_type,
        temperature_c=temperature_c,
        humidity_percent=humidity_percent,
        sample_spacing_m=sample_spacing_m,
    )
