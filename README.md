# SoilScan Sentinel-2 API

A FastAPI backend that accepts a GIS polygon or bounding box, queries locally stored Sentinel-2 satellite imagery and SoilGrids soil property data, and returns soil nutrient predictions using trained machine learning models.

**Live API:** `https://soilscan-sentinel2-api-production.up.railway.app`
**Interactive docs:** `https://soilscan-sentinel2-api-production.up.railway.app/docs`

## What it predicts

| Target | Classes | Model |
|--------|---------|-------|
| Nitrogen (N) | Low / Medium / High | Random Forest |
| Phosphorus (P) | Low / Medium / High | Random Forest |
| Potassium (K) | Low / Medium / High | SVM (RBF) |
| pH | 4.0 – 7.6 (11-class CPR scale) | Random Forest |

## How it works

### Step 1 — Polygon → grid of sample points

The input polygon (GeoJSON or bounding box) is projected to UTM and filled with a regular grid of points at 10 m spacing (matching Sentinel-2 native resolution). Only points that fall **inside** the polygon boundary are kept.

```
Polygon boundary
┌─────────────────┐
│  · · · · · · ·  │
│  · · · · · · ·  │  ← each · is a (lon, lat) point 10 m apart
│  · · · · · · ·  │
└─────────────────┘
```

A 1 hectare field produces ~100 sample points. The maximum is capped at 500 points per request (configurable via `SOILSCAN_MAX_SAMPLE_POINTS`).

---

### Step 2 — Each point → spectral band values

For every sample point the extractor performs a coordinate-to-pixel lookup against the local Sentinel-2 GeoTIFF:

1. Transform `(lon, lat)` from WGS84 → raster CRS (UTM Zone 51N)
2. Convert the UTM coordinate to a pixel `(row, col)` index using rasterio
3. Read a **3×3 pixel window** (30×30 m neighbourhood) centred on that pixel
4. Take `nanmean` across the 9 pixels as the band value for that point

```
Sentinel-2 raster (10 m pixels)
┌───┬───┬───┬───┬───┐
│   │   │   │   │   │
├───┼───┼───┼───┼───┤
│   │ █ │ █ │ █ │   │
├───┼───┼───┼───┼───┤  ← 3×3 window read around the matched pixel
│   │ █ │ ✦ │ █ │   │  ✦ = sample point projected to raster CRS
├───┼───┼───┼───┼───┤
│   │ █ │ █ │ █ │   │
├───┼───┼───┼───┼───┤
│   │   │   │   │   │
└───┴───┴───┴───┴───┘
band_value = nanmean(9 pixels)
```

This produces a `(N, 12)` array of band means and a `(N, 12)` array of temporal standard deviations across tiles — 24 spectral features total.

---

### Step 3 — Each point → SoilGrids priors

The same coordinate-to-pixel lookup is applied to locally stored SoilGrids v2 GeoTIFFs (250 m resolution). Six soil properties at two depths (0–5 cm, 5–15 cm):

| Property | Unit | What it captures |
|----------|------|-----------------|
| `phh2o` | pH | Soil acidity / alkalinity |
| `soc` | dg/kg | Soil organic carbon |
| `nitrogen` | cg/kg | Total nitrogen stock |
| `clay` | g/kg | Clay particle fraction |
| `sand` | g/kg | Sand particle fraction |
| `cec` | mmol/kg | Cation exchange capacity |

This gives 12 SoilGrids features per point (`sg_{property}_{depth}`).

---

### Step 4 — Each point → terrain features

A local DEM GeoTIFF is sampled at each point to extract 7 terrain attributes via numpy gradients on an 11×11 pixel window. If `dem.tif` is absent, the API automatically downloads the SRTM 30 m tile from AWS public S3 and saves it to the Volume permanently. If that fails, it falls back to the Open-Elevation API for elevation only.

| Feature | Description |
|---------|-------------|
| `elevation_m` | Elevation above sea level |
| `slope_deg` | Steepness of terrain |
| `aspect_deg` | Direction the slope faces (0=North, clockwise) |
| `twi` | Topographic Wetness Index — proxy for soil moisture accumulation |
| `curvature` | Surface concavity/convexity |
| `northness` | cos(aspect) — how north-facing the slope is |
| `eastness` | sin(aspect) — how east-facing the slope is |

---

### Step 5 — Spectral indices computed on the fly

Ten spectral indices are derived from the raw band values at each point:

| Index | Formula | Captures |
|-------|---------|---------|
| NDVI | (B08−B04)/(B08+B04) | Vegetation density |
| EVI | 2.5×(B08−B04)/(B08+6×B04−7.5×B02+1) | Canopy greenness (soil-adjusted) |
| SAVI | 1.5×(B08−B04)/(B08+B04+0.5) | Vegetation with soil correction |
| MSAVI | (2×B08+1−√((2×B08+1)²−8×(B08−B04)))/2 | Modified soil adjustment |
| NDRE | (B8A−B05)/(B8A+B05) | Chlorophyll / nitrogen stress |
| CHL-re | (B8A/B05)−1 | Canopy chlorophyll content |
| BSI | ((B11+B04)−(B08+B02))/((B11+B04)+(B08+B02)) | Bare soil exposure |
| BI | √((B04²+B08²)/2) | Overall surface brightness |
| NDWI | (B03−B08)/(B03+B08) | Surface water / moisture |
| NDMI | (B08−B11)/(B08+B11) | Dry matter / canopy water |

---

### Step 6 — Feature assembly (57 features per point)

```
[ B01…B12 (12) ]  +  [ B01_std…B12_std (12) ]  +  [ temp, humidity, altitude (3) ]
+  [ elevation…eastness (7) ]  +  [ sg_phh2o…sg_cec (12) ]
+  [ NDVI…NDMI (10) ]  +  [ crop_type (1, one-hot encoded inside pipeline) ]
= 57 input features
```

The sklearn Pipeline embedded in each `.joblib` model handles StandardScaler normalisation and OneHotEncoding automatically — no manual preprocessing needed at inference time.

---

### Step 7 — Inference and aggregation

Each of the four models runs independently on all N sample points:

```
point_1 → Low N,  Medium P,  Low K,  pH 6.4
point_2 → Low N,  Medium P,  Low K,  pH 6.0
point_3 → Low N,  High P,    Low K,  pH 6.4
   ...
─────────────────────────────────────────────────────────────
polygon → dominant: Low N · Medium P · Low K · pH 6.4
          distribution: N={Low:1.0} P={Low:0.1, Medium:0.67, High:0.33} ...
```

The response includes:
- **`dominant_class`** — majority prediction across all points
- **`class_distribution`** — fraction of points per class (spatial variability within the field)
- **`mean_probability`** — average model confidence per class

---

## API reference

### `GET /health`

```
GET /health
→ { "status": "ok" }
```

---

### `GET /predict` — bounding box

```
GET /predict?minlon=120.590&minlat=16.455&maxlon=120.600&maxlat=16.465&crop_type=cabbage
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `minlon` | float | yes | — | West boundary longitude |
| `minlat` | float | yes | — | South boundary latitude |
| `maxlon` | float | yes | — | East boundary longitude |
| `maxlat` | float | yes | — | North boundary latitude |
| `crop_type` | string | no | `"unknown"` | e.g. `cabbage`, `tomato`, `potato` |
| `temperature_c` | float | no | `18.0` | Air temperature in °C |
| `humidity_percent` | float | no | `80.0` | Relative humidity % |
| `sample_spacing_m` | float | no | `10.0` | Grid spacing in metres (5–100) |

---

### `POST /predict` — GeoJSON polygon

```json
{
  "polygon": {
    "type": "Polygon",
    "coordinates": [
      [[120.596, 16.462], [120.608, 16.462], [120.608, 16.471], [120.596, 16.471], [120.596, 16.462]]
    ]
  },
  "crop_type": "cabbage",
  "temperature_c": 18.0,
  "humidity_percent": 80.0,
  "sample_spacing_m": 10.0
}
```

---

### Response (both endpoints)

```json
{
  "nitrogen":   { "dominant_class": "Low (<11 mg/kg)", "class_distribution": {...}, "mean_probability": {...} },
  "phosphorus": { "dominant_class": "High (>25 mg/kg)", "class_distribution": {...}, "mean_probability": {...} },
  "potassium":  { "dominant_class": "Medium (78-156 mg/kg)", "class_distribution": {...}, "mean_probability": {...} },
  "ph":         { "dominant_class": "6.0", "class_distribution": {...}, "mean_probability": {...} },
  "sample_count": 143,
  "polygon_area_ha": 1.43,
  "warnings": []
}
```

| Code | Meaning |
|------|---------|
| `422` | Invalid polygon or bbox |
| `503` | Sentinel-2 data not found |

---

## Deploying to Railway

### 1. Connect the GitHub repo

**New Project → Deploy from GitHub repo** → select this repo. Railway builds via `Dockerfile`.

### 2. Create a Volume

New → Volume → mount path `/mnt/soilscan-data` → attach to service.

### 3. Set environment variables

| Variable | Value |
|----------|-------|
| `SOILSCAN_SENTINEL2_DIR` | `/mnt/soilscan-data/sentinel2` |
| `SOILSCAN_SOILGRIDS_DIR` | `/mnt/soilscan-data/soilgrids` |
| `SOILSCAN_DEM_PATH` | `/mnt/soilscan-data/dem/dem.tif` |
| `SOILSCAN_ADMIN_TOKEN` | `<your-secret-token>` |

### 4. Upload data files via admin endpoints

All admin endpoints require the `X-Admin-Token` header.

**Upload preprocessed Sentinel-2 files** (Google Drive or direct URL):
```http
POST /admin/download
X-Admin-Token: <token>
{ "url": "<drive-link>", "target": "bands_mean" }
{ "url": "<drive-link>", "target": "bands_std" }
```

**Upload SoilGrids as a zip:**
```http
POST /admin/unzip
X-Admin-Token: <token>
{ "url": "<drive-link>", "dest_dir": "soilgrids" }
```

Then fix any Windows path issues (if zip was created on Windows):
```http
POST /admin/fix-paths
X-Admin-Token: <token>
```

**DEM is auto-downloaded** on the first predict request — no manual upload needed.

**Check what's on the Volume:**
```http
GET /admin/files
GET /admin/ls
```

### Preprocessing Sentinel-2 data locally

The raw `.SAFE` tiles (~GB each) must be preprocessed into compact GeoTIFFs before upload:

```bash
python scripts/preprocess_sentinel2.py \
    --safe-dir D:/path/to/SAFE/tiles \
    --out-dir  data/sentinel2 \
    --aoi 120.3 16.2 120.85 16.85

python scripts/clip_sentinel2.py \
    --in-dir  data/sentinel2 \
    --out-dir data/sentinel2_clipped
```

Upload `data/sentinel2_clipped/bands_mean.tif` and `bands_std.tif` to Google Drive, then use `POST /admin/download`.

---

## Local setup

```bash
pip install -r requirements.txt
hypercorn main:app --reload
# API docs: http://localhost:8000/docs
```

Place data files at `data/sentinel2/`, `data/soilgrids/`, `data/dem/` or set the `SOILSCAN_*` env vars.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SOILSCAN_SENTINEL2_DIR` | `data/sentinel2` | Path to preprocessed S2 GeoTIFFs |
| `SOILSCAN_SOILGRIDS_DIR` | `data/soilgrids` | Path to SoilGrids GeoTIFFs |
| `SOILSCAN_DEM_PATH` | `data/dem/dem.tif` | Path to DEM GeoTIFF |
| `SOILSCAN_MODELS_DIR` | `models` | Path to .joblib model files |
| `SOILSCAN_MAX_SAMPLE_POINTS` | `500` | Cap on grid points per request |
| `SOILSCAN_DEFAULT_TEMPERATURE_C` | `18.0` | Fallback air temperature (°C) |
| `SOILSCAN_DEFAULT_HUMIDITY_PERCENT` | `80.0` | Fallback relative humidity (%) |
| `SOILSCAN_ADMIN_TOKEN` | _(unset)_ | Token for `/admin/*` endpoints |
