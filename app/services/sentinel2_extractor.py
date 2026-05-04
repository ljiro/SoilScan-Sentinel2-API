"""
Extract Sentinel-2 L2A band values from locally available data.

Two modes (auto-detected, fastest first):

  1. Preprocessed GeoTIFFs (preferred for deployment)
     Place in sentinel2_dir:
         bands_mean.tif  — 12-band stacked GeoTIFF (mean across tiles, 10 m)
         bands_std.tif   — 12-band stacked GeoTIFF (temporal std across tiles)
     Generate with: python scripts/preprocess_sentinel2.py

  2. Raw .SAFE tiles (fallback, for local development)
     Scans GRANULE/*/IMG_DATA/{R10m,R20m,R60m}/*.jp2 and samples on the fly.
     Much slower and requires GB-scale tile data.

In both modes the same 3×3 pixel neighbourhood (nanmean) sampling is used,
matching data_fetcher_copernicus.py from the training pipeline exactly.
"""
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from pyproj import Transformer

BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
_BAND_RE = re.compile(r"_(B(?:0[1-9]|1[0-2]|8A))_")
_RES_DIRS = ["R10m", "R20m", "R60m"]


# ---------------------------------------------------------------------------
# Mode 1: preprocessed GeoTIFFs
# ---------------------------------------------------------------------------

def _sample_stacked_geotiff(
    tif_path: Path,
    points: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Sample all 12 bands from a stacked GeoTIFF at each point using a 3×3 window.
    Returns (N, 12) float array; NaN for out-of-bounds points.
    """
    n = len(points)
    result = np.full((n, len(BAND_NAMES)), np.nan)
    try:
        with rasterio.open(tif_path) as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            nodata = src.nodata
            for i, (lon, lat) in enumerate(points):
                try:
                    x, y = transformer.transform(lon, lat)
                    row, col = src.index(x, y)
                    win = rasterio.windows.Window(col - 1, row - 1, 3, 3)
                    patch = src.read(window=win).astype(float)  # (12, 3, 3)
                    if nodata is not None:
                        patch[patch == nodata] = np.nan
                    result[i] = np.nanmean(patch, axis=(1, 2))  # (12,)
                except Exception:
                    pass
    except Exception:
        pass
    return result


def _extract_from_geotiffs(
    sentinel2_dir: Path,
    points: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    mean_path = sentinel2_dir / "bands_mean.tif"
    std_path  = sentinel2_dir / "bands_std.tif"
    means = _sample_stacked_geotiff(mean_path, points)
    stds  = (
        _sample_stacked_geotiff(std_path, points)
        if std_path.exists()
        else np.full_like(means, np.nan)
    )
    return means, stds


# ---------------------------------------------------------------------------
# Mode 2: raw .SAFE tiles (fallback)
# ---------------------------------------------------------------------------

def _find_safe_dirs(sentinel2_dir: Path) -> List[Path]:
    return sorted(sentinel2_dir.glob("*.SAFE"))


def _find_band_files(safe_dir: Path) -> Dict[str, Path]:
    band_files: Dict[str, Path] = {}
    for res in _RES_DIRS:
        pattern = str(safe_dir / "GRANULE" / "*" / "IMG_DATA" / res / "*.jp2")
        for path in glob.glob(pattern):
            m = _BAND_RE.search(Path(path).name)
            if m:
                band_name = m.group(1)
                if band_name not in band_files:
                    band_files[band_name] = Path(path)
    return band_files


def _sample_safe_tile(
    band_files: Dict[str, Path],
    points: List[Tuple[float, float]],
) -> np.ndarray:
    n = len(points)
    result = np.full((n, len(BAND_NAMES)), np.nan)
    for j, band in enumerate(BAND_NAMES):
        if band not in band_files:
            continue
        try:
            with rasterio.open(band_files[band]) as src:
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
                        result[i, j] = float(np.nanmean(patch))
                    except Exception:
                        pass
        except Exception:
            pass
    return result


def _extract_from_safe(
    sentinel2_dir: Path,
    points: List[Tuple[float, float]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    safe_dirs = _find_safe_dirs(sentinel2_dir)
    if not safe_dirs:
        return None, None
    tile_arrays = []
    for safe_dir in safe_dirs:
        band_files = _find_band_files(safe_dir)
        if band_files:
            tile_arrays.append(_sample_safe_tile(band_files, points))
    if not tile_arrays:
        return None, None
    stacked = np.stack(tile_arrays, axis=0)
    means = np.nanmean(stacked, axis=0)
    stds  = np.nanstd(stacked, axis=0) if len(tile_arrays) > 1 else np.full_like(means, np.nan)
    return means, stds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_bands(
    sentinel2_dir: Path,
    points: List[Tuple[float, float]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract S2 band values for all points.

    Returns:
        means — (N, 12) mean reflectance per band
        stds  — (N, 12) temporal std per band (NaN if single tile / not available)
    Returns (None, None) if no data source is found.
    """
    # Prefer preprocessed GeoTIFFs (fast, compact)
    if (sentinel2_dir / "bands_mean.tif").exists():
        means, stds = _extract_from_geotiffs(sentinel2_dir, points)
        if not np.all(np.isnan(means)):
            return means, stds

    # Fall back to raw .SAFE tiles
    return _extract_from_safe(sentinel2_dir, points)
