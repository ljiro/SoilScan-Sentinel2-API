from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    polygon: Dict[str, Any] = Field(
        ...,
        description="GeoJSON Polygon or MultiPolygon geometry object",
        example={
            "type": "Polygon",
            "coordinates": [
                [[120.5, 16.4], [120.51, 16.4], [120.51, 16.41], [120.5, 16.41], [120.5, 16.4]]
            ],
        },
    )
    crop_type: str = Field(
        default="unknown",
        description="Crop type at the field (e.g. cabbage, tomato, potato)",
    )
    temperature_c: Optional[float] = Field(
        default=None,
        description="Air temperature in °C at time of inference. Defaults to Benguet highland median.",
    )
    humidity_percent: Optional[float] = Field(
        default=None,
        description="Relative humidity (%) at time of inference. Defaults to Benguet highland median.",
    )
    sample_spacing_m: float = Field(
        default=10.0,
        ge=5.0,
        le=100.0,
        description="Grid sampling spacing in metres. Smaller = more points, higher accuracy, slower.",
    )


class NutrientPrediction(BaseModel):
    dominant_class: str = Field(description="Most common predicted class across sampled points")
    class_distribution: Dict[str, float] = Field(
        description="Fraction of sampled points assigned to each class"
    )
    mean_probability: Dict[str, float] = Field(
        description="Mean predicted probability per class across all sampled points"
    )


class PredictResponse(BaseModel):
    nitrogen: NutrientPrediction
    phosphorus: NutrientPrediction
    potassium: NutrientPrediction
    ph: NutrientPrediction
    sample_count: int = Field(description="Number of 10 m grid points sampled inside the polygon")
    polygon_area_ha: float = Field(description="Polygon area in hectares")
    warnings: List[str] = Field(default_factory=list)
