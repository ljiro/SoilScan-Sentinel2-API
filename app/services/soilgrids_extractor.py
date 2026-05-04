"""
Extract SoilGrids v2 priors from locally downloaded GeoTIFF or VRT files.

Expected directory layout (mirrors fetch_soilgrids.py download structure):
    data/soilgrids/
        phh2o/phh2o_0-5cm_mean.tif   (or .vrt)
        phh2o/phh2o_5-15cm_mean.tif
        soc/soc_0-5cm_mean.tif
        ...

Divisors applied to convert stored integer DN to real-unit values:
    phh2o    ÷ 10   → pH units
    soc      ÷ 10   → dg/kg
    nitrogen ÷ 100  → cg/kg
    clay     ÷ 10   → g/kg
    sand     ÷ 10   → g/kg
    cec      ÷ 10   → mmol(c)/kg

Falls back to the SoilGrids REST API when a local file is not found.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import rasterio
from pyproj import Transformer

PROPERTIES = ["phh2o", "soc", "nitrogen", "clay", "sand", "cec"]
DEPTHS = ["0-5cm", "5-15cm"]

_D_FACTOR: Dict[str, float] = {
    "phh2o": 10,
    "soc": 10,
    "nitrogen": 100,
    "clay": 10,
    "sand": 10,
    "cec": 10,
}

_REST_URL = "https://api.isric.org/soilgrids/v2.0/properties/query"
_REST_URL_FALLBACK = "https://rest.soilgrids.org/soilgrids/v2.0/properties/query"

# Output column names match training: sg_{prop}_{depth}
SG_COLUMNS = [f"sg_{p}_{d}" for p in PROPERTIES for d in DEPTHS]


def _local_path(soilgrids_dir: Path, prop: str, depth: str) -> Optional[Path]:
    for ext in (".tif", ".vrt"):
        p = soilgrids_dir / prop / f"{prop}_{depth}_mean{ext}"
        if p.exists():
            return p
    return None


def _sample_raster(path: Path, points: List[Tuple[float, float]]) -> np.ndarray:
    """Sample a single raster at all points. Returns 1-D array of floats (NaN on miss)."""
    values = np.full(len(points), np.nan)
    try:
        with rasterio.open(path) as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            nodata = src.nodata
            for i, (lon, lat) in enumerate(points):
                try:
                    x, y = transformer.transform(lon, lat)
                    row, col = src.index(x, y)
                    win = rasterio.windows.Window(col - 1, row - 1, 3, 3)
                    patch = src.read(1, window=win).astype(float)
                    if nodata is not None:
                        patch[patch == nodata] = np.nan
                    values[i] = float(np.nanmean(patch))
                except Exception:
                    pass
    except Exception:
        pass
    return values


def _fetch_rest(lon: float, lat: float) -> Dict[str, float]:
    """Fetch a single point from SoilGrids REST API. Returns {sg_col: value}."""
    params = {
        "lon": lon,
        "lat": lat,
        "property": PROPERTIES,
        "depth": DEPTHS,
        "value": "mean",
    }
    for url in (_REST_URL, _REST_URL_FALLBACK):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            layers = resp.json()["properties"]["layers"]
            result: Dict[str, float] = {}
            for layer in layers:
                prop = layer["name"]
                for depth_obj in layer["depths"]:
                    depth = depth_obj["label"]
                    raw = depth_obj["values"].get("mean")
                    if raw is not None and prop in _D_FACTOR:
                        col = f"sg_{prop}_{depth}"
                        result[col] = raw / _D_FACTOR[prop]
            return result
        except Exception:
            continue
    return {}


def extract_soilgrids(
    soilgrids_dir: Path,
    points: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Returns (N, 12) array of SoilGrids values in SG_COLUMNS order.
    Priority: local GeoTIFF/VRT → REST API → NaN.
    """
    n = len(points)
    result = np.full((n, len(SG_COLUMNS)), np.nan)

    col_idx = {col: i for i, col in enumerate(SG_COLUMNS)}

    # Collect which columns still need REST fallback per point
    rest_needed = [False] * n

    for prop in PROPERTIES:
        for depth in DEPTHS:
            col = f"sg_{prop}_{depth}"
            ci = col_idx[col]
            local = _local_path(soilgrids_dir, prop, depth)
            if local:
                raw = _sample_raster(local, points)
                result[:, ci] = raw / _D_FACTOR[prop]
            else:
                rest_needed = [True] * n  # mark all for REST for this column
                break
        else:
            continue
        break

    # REST fallback: fill any still-NaN points
    for i, (lon, lat) in enumerate(points):
        if np.any(np.isnan(result[i])):
            fetched = _fetch_rest(lon, lat)
            for col, val in fetched.items():
                if col in col_idx:
                    ci = col_idx[col]
                    if np.isnan(result[i, ci]):
                        result[i, ci] = val

    return result
