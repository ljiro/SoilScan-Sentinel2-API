# SoilScan Sentinel-2 API

A FastAPI backend that accepts a GIS polygon, queries locally downloaded Sentinel-2 satellite imagery and SoilGrids soil property data, and returns soil nutrient predictions using trained machine learning models.

## What it predicts

| Target | Classes | Model |
|--------|---------|-------|
| Nitrogen (N) | Low / Medium / High | Random Forest |
| Phosphorus (P) | Low / Medium / High | Random Forest |
| Potassium (K) | Low / Medium / High | SVM (RBF) |
| pH | 4.0 – 7.6 (11-class CPR scale) | Random Forest |

## How it works

1. **Polygon sampling** — generates a regular 10 m grid of points inside the input polygon
2. **Sentinel-2 extraction** — samples all 12 spectral bands (B01–B12, B8A) from local `.SAFE` tiles using a 3×3 pixel neighbourhood
3. **SoilGrids extraction** — reads 12 soil prior columns (pH, SOC, N, clay, sand, CEC at 0–5 cm and 5–15 cm depths) from local GeoTIFF/VRT files
4. **Terrain extraction** — derives elevation, slope, aspect, TWI, curvature, northness, eastness from a local DEM
5. **Spectral indices** — computes NDVI, EVI, SAVI, MSAVI, NDRE, CHL-re, BSI, BI, NDWI, NDMI on the fly
6. **Inference** — assembles a 57-feature DataFrame and runs it through four sklearn Pipeline models
7. **Aggregation** — majority-votes predictions across all sampled points and returns class distributions

## Project structure

```
├── main.py                        # FastAPI app entry point
├── app/
│   ├── core/config.py             # Settings (configurable via env vars)
│   ├── api/predict.py             # POST /predict route
│   ├── schemas/predict.py         # Pydantic request / response models
│   └── services/
│       ├── polygon_sampler.py     # UTM grid sampling from GeoJSON polygon
│       ├── sentinel2_extractor.py # S2 band extraction from .SAFE tiles
│       ├── soilgrids_extractor.py # SoilGrids GeoTIFF/VRT reader + REST fallback
│       ├── terrain_extractor.py   # DEM terrain attribute extraction
│       ├── spectral_indices.py    # Spectral index computation
│       └── predictor.py           # Model loading, feature assembly, inference
├── data/
│   ├── sentinel2/                 # Place .SAFE tile directories here
│   ├── soilgrids/                 # Place SoilGrids GeoTIFF/VRT files here
│   └── dem/                       # Place DEM GeoTIFF here
├── models/                        # Place .joblib + _meta.json model files here
└── scripts/
    └── preprocess_terrain.py      # One-time terrain raster generation from DEM
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add model files

Copy the trained model files from the training repository into `models/`:

```
models/
├── n_RandomForest.joblib
├── n_RandomForest_meta.json
├── p_RandomForest.joblib
├── p_RandomForest_meta.json
├── k_SVM.joblib
├── k_SVM_meta.json
├── ph_RandomForest.joblib
└── ph_RandomForest_meta.json
```

### 3. Add data files

**Sentinel-2** — place downloaded `.SAFE` directories inside `data/sentinel2/`:
```
data/sentinel2/
└── S2B_MSIL2A_20251101T....SAFE/
    └── GRANULE/*/IMG_DATA/{R10m,R20m,R60m}/*.jp2
```

**SoilGrids** — place GeoTIFF or VRT files at:
```
data/soilgrids/
├── phh2o/phh2o_0-5cm_mean.tif
├── phh2o/phh2o_5-15cm_mean.tif
├── soc/soc_0-5cm_mean.tif
├── soc/soc_5-15cm_mean.tif
├── nitrogen/nitrogen_0-5cm_mean.tif
├── nitrogen/nitrogen_5-15cm_mean.tif
├── clay/clay_0-5cm_mean.tif
├── clay/clay_5-15cm_mean.tif
├── sand/sand_0-5cm_mean.tif
├── sand/sand_5-15cm_mean.tif
├── cec/cec_0-5cm_mean.tif
└── cec/cec_5-15cm_mean.tif
```

**DEM** — place a GeoTIFF at `data/dem/dem.tif`.

Optionally pre-compute terrain rasters for better TWI accuracy (requires `richdem`):
```bash
python scripts/preprocess_terrain.py --dem data/dem/dem.tif
```

### 4. Run the server

```bash
hypercorn main:app --reload
```

The interactive API docs are available at `http://localhost:8000/docs`.

## API reference

### `POST /predict`

**Request body**

```json
{
  "polygon": {
    "type": "Polygon",
    "coordinates": [[[120.5, 16.4], [120.51, 16.4], [120.51, 16.41], [120.5, 16.41], [120.5, 16.4]]]
  },
  "crop_type": "cabbage",
  "temperature_c": 18.0,
  "humidity_percent": 80.0,
  "sample_spacing_m": 10.0
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `polygon` | GeoJSON Geometry | required | Polygon or MultiPolygon |
| `crop_type` | string | `"unknown"` | Crop at the field (cabbage, tomato, potato, …) |
| `temperature_c` | float | 18.0 | Air temperature in °C |
| `humidity_percent` | float | 80.0 | Relative humidity % |
| `sample_spacing_m` | float 5–100 | 10.0 | Grid spacing in metres |

**Response**

```json
{
  "nitrogen": {
    "dominant_class": "Low",
    "class_distribution": {"Low": 0.72, "Medium": 0.28, "High": 0.0},
    "mean_probability": {"Low": 0.68, "Medium": 0.29, "High": 0.03}
  },
  "phosphorus": { "..." },
  "potassium": { "..." },
  "ph": {
    "dominant_class": "6.4",
    "class_distribution": {"4.0": 0.0, "6.4": 0.61, "6.8": 0.39, "...": 0.0},
    "mean_probability": { "..." }
  },
  "sample_count": 143,
  "polygon_area_ha": 1.43,
  "warnings": []
}
```

### `GET /health`

Returns `{"status": "ok"}`.

## Configuration

All settings can be overridden with environment variables (prefix `SOILSCAN_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SOILSCAN_SENTINEL2_DIR` | `data/sentinel2` | Path to .SAFE tile directory |
| `SOILSCAN_SOILGRIDS_DIR` | `data/soilgrids` | Path to SoilGrids files |
| `SOILSCAN_DEM_PATH` | `data/dem/dem.tif` | Path to DEM GeoTIFF |
| `SOILSCAN_MODELS_DIR` | `models` | Path to .joblib model files |
| `SOILSCAN_MAX_SAMPLE_POINTS` | `500` | Cap on grid points per request |
| `SOILSCAN_DEFAULT_SAMPLE_SPACING_M` | `10.0` | Default grid spacing |
| `SOILSCAN_DEFAULT_TEMPERATURE_C` | `18.0` | Fallback air temperature |
| `SOILSCAN_DEFAULT_HUMIDITY_PERCENT` | `80.0` | Fallback relative humidity |
