"""
Extract Sentinel-2 L2A band values from locally downloaded .SAFE tiles.

Replicates the exact sampling logic from SoilScan-Sentinel2/src/data_fetcher_copernicus.py:
  - Scans GRANULE/*/IMG_DATA/{R10m,R20m,R60m}/*.jp2
  - Samples a 3×3 pixel neighbourhood (nanmean) around each point
  - Computes mean and std across all tiles covering a point
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


def _sample_tile(
    band_files: Dict[str, Path],
    points: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Sample all bands at all points for one .SAFE tile.
    Returns (N, 12) array; NaN where point falls outside tile or band missing.
    """
    n = len(points)
    result = np.full((n, len(BAND_NAMES)), np.nan)

    for j, band in enumerate(BAND_NAMES):
        if band not in band_files:
            continue
        path = band_files[band]
        try:
            with rasterio.open(path) as src:
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                nodata = src.nodata
                for i, (lon, lat) in enumerate(points):
                    try:
                        x, y = transformer.transform(lon, lat)
                        row, col = src.index(x, y)
                        # 3×3 window
                        win = rasterio.windows.Window(col - 1, row - 1, 3, 3)
                        patch = src.read(1, window=win).astype(float)
                        if nodata is not None:
                            patch[patch == nodata] = np.nan
                        val = float(np.nanmean(patch))
                        result[i, j] = val
                    except Exception:
                        # Point outside tile bounds — leave as NaN
                        pass
        except Exception:
            pass

    return result


def extract_bands(
    sentinel2_dir: Path,
    points: List[Tuple[float, float]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract S2 band values for all points from all .SAFE tiles in sentinel2_dir.

    Returns:
        means  — (N, 12) mean reflectance per band (averaged across tiles)
        stds   — (N, 12) temporal std per band (across tiles, NaN if single tile)
    Returns (None, None) if no .SAFE tiles are found.
    """
    safe_dirs = _find_safe_dirs(sentinel2_dir)
    if not safe_dirs:
        return None, None

    tile_arrays: List[np.ndarray] = []
    for safe_dir in safe_dirs:
        band_files = _find_band_files(safe_dir)
        if not band_files:
            continue
        arr = _sample_tile(band_files, points)
        tile_arrays.append(arr)

    if not tile_arrays:
        return None, None

    stacked = np.stack(tile_arrays, axis=0)          # (T, N, 12)
    means = np.nanmean(stacked, axis=0)              # (N, 12)
    stds = np.nanstd(stacked, axis=0) if len(tile_arrays) > 1 else np.full_like(means, np.nan)

    return means, stds
