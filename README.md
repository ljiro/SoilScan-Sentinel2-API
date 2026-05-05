# SoilScan Sentinel-2 API

A FastAPI backend that accepts a GIS polygon or bounding box, queries locally downloaded Sentinel-2 satellite imagery and SoilGrids soil property data, and returns soil nutrient predictions using trained machine learning models.

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
├── nixpacks.toml                  # Railway build config (GDAL/PROJ/GEOS system deps)
├── railway.json                   # Railway start command
├── app/
│   ├── core/config.py             # Settings (configurable via env vars)
│   ├── api/predict.py             # GET + POST /predict routes
│   ├── schemas/predict.py         # Pydantic request / response models
│   └── services/
│       ├── polygon_sampler.py     # UTM grid sampling from GeoJSON polygon
│       ├── sentinel2_extractor.py # S2 band extraction from .SAFE tiles
│       ├── soilgrids_extractor.py # SoilGrids GeoTIFF/VRT reader + REST fallback
│       ├── terrain_extractor.py   # DEM terrain attribute extraction
│       ├── spectral_indices.py    # Spectral index computation
│       └── predictor.py           # Model loading, feature assembly, inference
├── models/                        # Trained .joblib pipelines + _meta.json (in git)
├── data/
│   ├── sentinel2/                 # .SAFE tile directories (Railway Volume)
│   ├── soilgrids/                 # SoilGrids GeoTIFF/VRT files (Railway Volume)
│   └── dem/                       # DEM GeoTIFF (Railway Volume)
└── scripts/
    └── preprocess_terrain.py      # One-time terrain raster generation from DEM
```

## Deploying to Railway

The repo is ready to deploy directly from GitHub. Railway auto-builds on every push using Nixpacks (GDAL, PROJ, GEOS system packages are declared in `nixpacks.toml`). The trained models are already committed to `models/`.

The only setup required is a **persistent Volume** for the large geospatial data files that cannot live in git.

### 1. Connect the GitHub repo

In the Railway dashboard: **New Project → Deploy from GitHub repo** → select this repo.

### 2. Create a Volume for data files

1. Inside the project → **New** → **Volume**
2. Name it `soilscan-data`, mount path `/mnt/soilscan-data`
3. Attach it to the service

### 3. Upload data to the Volume

Open the Railway shell (service → **Shell** tab) and place your files at:

```
/mnt/soilscan-data/
├── sentinel2/
│   └── S2B_MSIL2A_....SAFE/
│       └── GRANULE/*/IMG_DATA/{R10m,R20m,R60m}/*.jp2
├── soilgrids/
│   ├── phh2o/phh2o_0-5cm_mean.tif
│   ├── phh2o/phh2o_5-15cm_mean.tif
│   ├── soc/soc_0-5cm_mean.tif
│   ├── soc/soc_5-15cm_mean.tif
│   ├── nitrogen/nitrogen_0-5cm_mean.tif
│   ├── nitrogen/nitrogen_5-15cm_mean.tif
│   ├── clay/clay_0-5cm_mean.tif
│   ├── clay/clay_5-15cm_mean.tif
│   ├── sand/sand_0-5cm_mean.tif
│   ├── sand/sand_5-15cm_mean.tif
│   ├── cec/cec_0-5cm_mean.tif
│   └── cec/cec_5-15cm_mean.tif
└── dem/
    └── dem.tif
```

### 4. Set environment variables

| Variable | Value |
|----------|-------|
| `SOILSCAN_SENTINEL2_DIR` | `/mnt/soilscan-data/sentinel2` |
| `SOILSCAN_SOILGRIDS_DIR` | `/mnt/soilscan-data/soilgrids` |
| `SOILSCAN_DEM_PATH` | `/mnt/soilscan-data/dem/dem.tif` |
| `SOILSCAN_MODELS_DIR` | `models` |

After that, every `git push` redeploys automatically with zero data loss — the Volume persists across deploys.

---

## Local setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add data files

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
...
```

**DEM** — place a GeoTIFF at `data/dem/dem.tif`.

Optionally pre-compute terrain rasters for more accurate TWI (requires `richdem`):
```bash
python scripts/preprocess_terrain.py --dem data/dem/dem.tif
```

### 3. Run the server

```bash
hypercorn main:app --reload
```

Interactive API docs: `http://localhost:8000/docs`

---

## API reference

Base URL: `https://your-domain.up.railway.app`
Interactive docs: `https://your-domain.up.railway.app/docs`

---

### `GET /health`

Health check. Use this to confirm the service is running.

```
GET /health
```

**Response**
```json
{ "status": "ok" }
```

---

### `GET /predict` — bounding box

Quick rectangular query. Useful for map-based clients and mobile apps.

```
GET /predict?minlon=120.50&minlat=16.40&maxlon=120.51&maxlat=16.41&crop_type=cabbage
```

**Query parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `minlon` | float | yes | — | West boundary longitude |
| `minlat` | float | yes | — | South boundary latitude |
| `maxlon` | float | yes | — | East boundary longitude |
| `maxlat` | float | yes | — | North boundary latitude |
| `crop_type` | string | no | `"unknown"` | Crop type e.g. `cabbage`, `tomato`, `potato` |
| `temperature_c` | float | no | `18.0` | Air temperature in °C |
| `humidity_percent` | float | no | `80.0` | Relative humidity % |
| `sample_spacing_m` | float | no | `10.0` | Grid sampling spacing in metres (5–100) |

**Example**
```
GET /predict?minlon=120.596&minlat=16.462&maxlon=120.608&maxlat=16.471&crop_type=cabbage&sample_spacing_m=20
```

---

### `POST /predict` — GeoJSON polygon

Exact field boundary query. Use this when you have a real field polygon from a GIS app.

```
POST /predict
Content-Type: application/json
```

**Request body**
```json
{
  "polygon": {
    "type": "Polygon",
    "coordinates": [
      [
        [120.596, 16.462],
        [120.608, 16.462],
        [120.608, 16.471],
        [120.596, 16.471],
        [120.596, 16.462]
      ]
    ]
  },
  "crop_type": "cabbage",
  "temperature_c": 18.0,
  "humidity_percent": 80.0,
  "sample_spacing_m": 10.0
}
```

**Fields**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `polygon` | GeoJSON Geometry | yes | — | `Polygon` or `MultiPolygon` in WGS84 |
| `crop_type` | string | no | `"unknown"` | Crop type e.g. `cabbage`, `tomato`, `potato` |
| `temperature_c` | float | no | `18.0` | Air temperature in °C |
| `humidity_percent` | float | no | `80.0` | Relative humidity % |
| `sample_spacing_m` | float | no | `10.0` | Grid sampling spacing in metres (5–100) |

---

### Response (both endpoints return the same shape)

```json
{
  "nitrogen": {
    "dominant_class": "Low",
    "class_distribution": {
      "Low": 0.72,
      "Medium": 0.28,
      "High": 0.0
    },
    "mean_probability": {
      "Low": 0.68,
      "Medium": 0.29,
      "High": 0.03
    }
  },
  "phosphorus": {
    "dominant_class": "Medium",
    "class_distribution": {"Low": 0.1, "Medium": 0.8, "High": 0.1},
    "mean_probability":   {"Low": 0.09, "Medium": 0.78, "High": 0.13}
  },
  "potassium": {
    "dominant_class": "Low",
    "class_distribution": {"Low": 0.65, "Medium": 0.35, "High": 0.0},
    "mean_probability":   {"Low": 0.61, "Medium": 0.37, "High": 0.02}
  },
  "ph": {
    "dominant_class": "6.4",
    "class_distribution": {
      "4.0": 0.0, "4.4": 0.0, "4.8": 0.0, "5.2": 0.0, "5.4": 0.0,
      "5.8": 0.0, "6.0": 0.12, "6.4": 0.61, "6.8": 0.27, "7.2": 0.0, "7.6": 0.0
    },
    "mean_probability": { "...": "..." }
  },
  "sample_count": 143,
  "polygon_area_ha": 1.43,
  "warnings": []
}
```

**Response fields**

| Field | Description |
|-------|-------------|
| `nitrogen` / `phosphorus` / `potassium` | N, P, K prediction — Low / Medium / High |
| `ph` | pH prediction on the 11-class CPR scale (4.0 – 7.6) |
| `dominant_class` | Most common predicted class across all sampled grid points |
| `class_distribution` | Fraction of grid points assigned to each class |
| `mean_probability` | Average model confidence per class across all grid points |
| `sample_count` | Number of 10 m grid points sampled inside the polygon |
| `polygon_area_ha` | Polygon area in hectares |
| `warnings` | Non-fatal issues e.g. sample count capped, DEM missing |

**HTTP error codes**

| Code | Meaning |
|------|---------|
| `422` | Invalid polygon or bbox |
| `503` | Sentinel-2 data not found on the server |

---

## Configuration

All settings can be overridden with environment variables (prefix `SOILSCAN_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SOILSCAN_SENTINEL2_DIR` | `data/sentinel2` | Path to .SAFE tile directory |
| `SOILSCAN_SOILGRIDS_DIR` | `data/soilgrids` | Path to SoilGrids files |
| `SOILSCAN_DEM_PATH` | `data/dem/dem.tif` | Path to DEM GeoTIFF |
| `SOILSCAN_MODELS_DIR` | `models` | Path to .joblib model files |
| `SOILSCAN_MAX_SAMPLE_POINTS` | `500` | Cap on grid points per request |
| `SOILSCAN_DEFAULT_SAMPLE_SPACING_M` | `10.0` | Default grid spacing in metres |
| `SOILSCAN_DEFAULT_TEMPERATURE_C` | `18.0` | Fallback air temperature (°C) |
| `SOILSCAN_DEFAULT_HUMIDITY_PERCENT` | `80.0` | Fallback relative humidity (%) |
