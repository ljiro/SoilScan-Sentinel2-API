"""
Load the four trained soil models and run batch inference over polygon sample points.

Models (joblib sklearn Pipelines with embedded ColumnTransformer preprocessing):
    n_RandomForest.joblib   → Nitrogen (Low / Medium / High)
    p_RandomForest.joblib   → Phosphorus (Low / Medium / High)
    k_SVM.joblib            → Potassium (Low / Medium / High)
    ph_RandomForest.joblib  → pH (4.0 … 7.6 on 11-class CPR scale)

Feature assembly order matches train_ordinal.py exactly:
    S2 bands (12) → S2 std (12) → microclimate (3) → terrain (7)
    → SoilGrids (12) → spectral indices (10) → crops (1 categorical)
    = 57 columns total (crops expands via OneHotEncoder inside the pipeline)
"""
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from app.schemas.predict import NutrientPrediction
from app.services.sentinel2_extractor import BAND_NAMES
from app.services.soilgrids_extractor import SG_COLUMNS
from app.services.spectral_indices import INDEX_NAMES, compute_indices
from app.services.terrain_extractor import TERRAIN_COLUMNS

# Canonical feature order (numeric columns only, before crops)
_BAND_STD_NAMES = [f"{b}_std" for b in BAND_NAMES]
_MICROCLIMATE = ["temperature_c", "humidity_percent", "altitude_m"]

_NUMERIC_COLS = (
    BAND_NAMES
    + _BAND_STD_NAMES
    + _MICROCLIMATE
    + TERRAIN_COLUMNS
    + SG_COLUMNS
    + INDEX_NAMES
)

_MODEL_FILES = {
    "nitrogen":   "n_RandomForest.joblib",
    "phosphorus": "p_RandomForest.joblib",
    "potassium":  "k_SVM.joblib",
    "ph":         "ph_RandomForest.joblib",
}


def load_models(models_dir: Path) -> Dict[str, Any]:
    """Load all four joblib pipelines and their _meta.json sidecars."""
    models: Dict[str, Any] = {}
    for target, filename in _MODEL_FILES.items():
        model_path = models_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                "Copy the .joblib files from SoilScan-Sentinel2/outputs/models/ into models/."
            )
        pipeline = joblib.load(model_path)
        meta_path = model_path.with_name(model_path.stem + "_meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        models[target] = {"pipeline": pipeline, "meta": meta}
    return models


def _build_feature_df(
    points: List[Tuple[float, float]],
    band_means: np.ndarray,    # (N, 12)
    band_stds: np.ndarray,     # (N, 12) — NaN if single tile
    soilgrids: np.ndarray,     # (N, 12)
    terrain: Optional[np.ndarray],  # (N, 7) or None
    temperature_c: float,
    humidity_percent: float,
    crop_type: str,
) -> pd.DataFrame:
    n = len(points)

    # Altitude from terrain elevation (same physical quantity)
    if terrain is not None:
        altitude = terrain[:, 0]  # elevation_m
    else:
        altitude = np.full(n, np.nan)

    microclimate = np.column_stack([
        np.full(n, temperature_c),
        np.full(n, humidity_percent),
        altitude,
    ])

    terrain_arr = terrain if terrain is not None else np.full((n, len(TERRAIN_COLUMNS)), np.nan)
    indices = compute_indices(band_means)

    numeric = np.column_stack([
        band_means,
        band_stds,
        microclimate,
        terrain_arr,
        soilgrids,
        indices,
    ])

    df = pd.DataFrame(numeric, columns=_NUMERIC_COLS)
    df["crops"] = crop_type

    return df


def _aggregate(
    pipeline: Any,
    df: pd.DataFrame,
    class_names: List[str],
) -> NutrientPrediction:
    """Run inference on all rows, aggregate to a polygon-level prediction."""
    preds = pipeline.predict(df)
    probas = pipeline.predict_proba(df)  # (N, C)

    # Dominant class: most frequent predicted label
    pred_labels = [class_names[int(p)] for p in preds]
    dominant = Counter(pred_labels).most_common(1)[0][0]

    # Class distribution: fraction of points per class
    label_counts = Counter(pred_labels)
    total = len(pred_labels)
    distribution = {cls: round(label_counts.get(cls, 0) / total, 4) for cls in class_names}

    # Mean probability per class across all points
    mean_proba = {
        cls: round(float(probas[:, i].mean()), 4) for i, cls in enumerate(class_names)
    }

    return NutrientPrediction(
        dominant_class=dominant,
        class_distribution=distribution,
        mean_probability=mean_proba,
    )


def predict_all(
    models: Dict[str, Any],
    points: List[Tuple[float, float]],
    band_means: np.ndarray,
    band_stds: np.ndarray,
    soilgrids: np.ndarray,
    terrain: Optional[np.ndarray],
    temperature_c: float,
    humidity_percent: float,
    crop_type: str,
) -> Dict[str, NutrientPrediction]:
    df = _build_feature_df(
        points=points,
        band_means=band_means,
        band_stds=band_stds,
        soilgrids=soilgrids,
        terrain=terrain,
        temperature_c=temperature_c,
        humidity_percent=humidity_percent,
        crop_type=crop_type,
    )

    # Align columns to what each pipeline was trained on
    results: Dict[str, NutrientPrediction] = {}
    for target, model_obj in models.items():
        pipeline = model_obj["pipeline"]
        meta = model_obj["meta"]
        class_names: List[str] = [str(c) for c in meta.get("class_names", [])]

        # Derive class_names from pipeline if meta is missing
        if not class_names:
            try:
                class_names = [str(c) for c in pipeline.classes_]
            except AttributeError:
                class_names = [str(c) for c in pipeline[-1].classes_]

        # Reorder / subset columns to match pipeline's fitted feature order
        try:
            ct = pipeline.steps[0][1]
            expected_cols = list(ct.feature_names_in_)
            df_in = df.reindex(columns=expected_cols)
        except AttributeError:
            df_in = df

        results[target] = _aggregate(pipeline, df_in, class_names)

    return results
