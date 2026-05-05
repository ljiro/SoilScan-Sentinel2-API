"""
Extract terrain features from a local DEM GeoTIFF.

Three modes (auto-detected, in priority order):

  1. Pre-computed rasters — fastest, most accurate.
     Place these next to dem.tif:
         data/dem/slope.tif  data/dem/aspect.tif
         data/dem/twi.tif    data/dem/curvature.tif

  2. On-the-fly computation from dem.tif using local numpy gradients.
     TWI is approximated as ln(1 / tan(slope)) — no flow accumulation.

  3. Auto-download SRTM tile + Open-Elevation API fallback.
     If dem.tif is missing, tries to download the SRTM HGT tile from a
     public AWS mirror, converts to GeoTIFF, saves permanently.
     If that also fails, fetches point elevations from the Open-Elevation API
     and returns elevation_m only (other features NaN).

Output columns (7):
    elevation_m, slope_deg, aspect_deg, twi, curvature, northness, eastness
"""
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from pyproj import Transformer

TERRAIN_COLUMNS = [
    "elevation_m", "slope_deg", "aspect_deg", "twi", "curvature", "northness", "eastness"
]


def _sample_raster_at_points(path: Path, points: List[Tuple[float, float]]) -> np.ndarray:
    """Sample a single raster at each point (center pixel). Returns 1-D float array."""
    values = np.full(len(points), np.nan)
    try:
        with rasterio.open(path) as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            nodata = src.nodata
            for i, (lon, lat) in enumerate(points):
                try:
                    x, y = transformer.transform(lon, lat)
                    row, col = src.index(x, y)
                    win = rasterio.windows.Window(col, row, 1, 1)
                    val = src.read(1, window=win).astype(float)[0, 0]
                    if nodata is not None and val == nodata:
                        val = np.nan
                    values[i] = val
                except Exception:
                    pass
    except Exception:
        pass
    return values


def _compute_from_dem(dem_path: Path, points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Compute terrain attributes on-the-fly from dem.tif.
    For each point reads a local 11×11 pixel window, computes gradient-based metrics.
    """
    n = len(points)
    result = np.full((n, len(TERRAIN_COLUMNS)), np.nan)
    _WINDOW = 11  # must be odd

    try:
        with rasterio.open(dem_path) as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            nodata = src.nodata
            half = _WINDOW // 2

            # Approximate cell size in metres at the centroid latitude
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)
            mid_lat = np.mean([lat for _, lat in points])
            cell_x_m = res_x * 111_320 * np.cos(np.radians(mid_lat))
            cell_y_m = res_y * 111_320

            for i, (lon, lat) in enumerate(points):
                try:
                    x, y = transformer.transform(lon, lat)
                    row, col = src.index(x, y)

                    win = rasterio.windows.Window(col - half, row - half, _WINDOW, _WINDOW)
                    dem_win = src.read(1, window=win).astype(float)
                    if nodata is not None:
                        dem_win[dem_win == nodata] = np.nan

                    cy, cx = half, half
                    elevation = dem_win[cy, cx]
                    if np.isnan(elevation):
                        continue

                    # Gradient (rise over run)
                    dy, dx = np.gradient(dem_win, cell_y_m, cell_x_m)

                    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
                    slope_deg = np.degrees(slope_rad)[cy, cx]

                    # Aspect: 0=North, clockwise
                    aspect_deg = (np.degrees(np.arctan2(-dx, dy)) % 360)[cy, cx]

                    # Curvature (Laplacian approximation)
                    d2x = np.gradient(dx, cell_x_m, axis=1)
                    d2y = np.gradient(dy, cell_y_m, axis=0)
                    curvature = -(d2x + d2y)[cy, cx]

                    # TWI approximation (no upslope area)
                    s = max(float(slope_rad[cy, cx]), 1e-4)
                    twi = float(np.log(1.0 / np.tan(s)))

                    northness = float(np.cos(np.radians(aspect_deg)))
                    eastness = float(np.sin(np.radians(aspect_deg)))

                    result[i] = [
                        float(elevation),
                        float(slope_deg),
                        float(aspect_deg),
                        twi,
                        float(curvature),
                        northness,
                        eastness,
                    ]
                except Exception:
                    pass
    except Exception:
        pass

    return result


def _srtm_tile_name(lat: float, lon: float) -> str:
    """Return the SRTM 1-arcsec HGT tile filename for a given lat/lon."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(int(lat)):02d}{ew}{abs(int(lon)):03d}.hgt"


def _download_srtm(dem_path: Path, lats: List[float], lons: List[float]) -> bool:
    """
    Try to download the SRTM HGT tile(s) covering the given points from the
    public AWS Terrain Tiles mirror and convert to a GeoTIFF at dem_path.
    Returns True on success.
    """
    try:
        import subprocess
        tiles_needed = set(
            _srtm_tile_name(lat, lon) for lat, lon in zip(lats, lons)
        )
        hgt_paths = []
        dem_path.parent.mkdir(parents=True, exist_ok=True)

        for tile in tiles_needed:
            ns_dir = tile[:3]  # e.g. "N16"
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns_dir}/{tile}.gz"
            gz_path = dem_path.parent / (tile + ".gz")
            hgt_path = dem_path.parent / tile
            try:
                urllib.request.urlretrieve(url, gz_path)
                subprocess.run(["gunzip", "-f", str(gz_path)], check=True, capture_output=True)
                if hgt_path.exists():
                    hgt_paths.append(hgt_path)
            except Exception:
                continue

        if not hgt_paths:
            return False

        # Merge tiles and write GeoTIFF using rasterio
        arrays, transforms, crss = [], [], []
        for hp in hgt_paths:
            with rasterio.open(hp) as src:
                arrays.append(src.read(1).astype(np.float32))
                transforms.append(src.transform)
                crss.append(src.crs)

        # Simple case: single tile
        arr = arrays[0]
        tf = transforms[0]
        crs = crss[0]

        with rasterio.open(
            dem_path, "w",
            driver="GTiff", height=arr.shape[0], width=arr.shape[1],
            count=1, dtype="float32", crs=crs, transform=tf,
            compress="deflate", nodata=-32768,
        ) as dst:
            dst.write(arr, 1)

        for hp in hgt_paths:
            hp.unlink(missing_ok=True)

        return True
    except Exception:
        return False


def _open_elevation_fallback(
    points: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Fetch elevation for each point from the Open-Elevation API.
    Returns (N, 7) array with elevation_m filled and other features NaN.
    """
    import json
    n = len(points)
    result = np.full((n, len(TERRAIN_COLUMNS)), np.nan)

    try:
        batch = [{"latitude": lat, "longitude": lon} for lon, lat in points]
        payload = json.dumps({"locations": batch}).encode()
        req = urllib.request.Request(
            "https://api.open-elevation.com/api/v1/lookup",
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        for i, r in enumerate(data.get("results", [])):
            elev = r.get("elevation")
            if elev is not None:
                result[i, 0] = float(elev)  # elevation_m
    except Exception:
        pass

    return result


def extract_terrain(
    dem_path: Path,
    points: List[Tuple[float, float]],
) -> Optional[np.ndarray]:
    """
    Returns (N, 7) array of terrain features in TERRAIN_COLUMNS order.
    If dem.tif is missing, attempts SRTM auto-download then Open-Elevation fallback.
    Returns None only if all sources fail.
    """
    if not dem_path.exists():
        lats = [lat for _, lat in points]
        lons = [lon for lon, _ in points]
        print("[terrain] DEM missing — attempting SRTM auto-download …")
        if _download_srtm(dem_path, lats, lons):
            print(f"[terrain] SRTM downloaded → {dem_path}")
        else:
            print("[terrain] SRTM download failed — using Open-Elevation API")
            return _open_elevation_fallback(points)

    dem_dir = dem_path.parent

    # Try pre-computed rasters first
    slope_p = dem_dir / "slope.tif"
    aspect_p = dem_dir / "aspect.tif"
    twi_p = dem_dir / "twi.tif"
    curv_p = dem_dir / "curvature.tif"

    if slope_p.exists() and aspect_p.exists() and twi_p.exists() and curv_p.exists():
        elevation = _sample_raster_at_points(dem_path, points)
        slope_deg = _sample_raster_at_points(slope_p, points)
        aspect_deg = _sample_raster_at_points(aspect_p, points)
        twi = _sample_raster_at_points(twi_p, points)
        curvature = _sample_raster_at_points(curv_p, points)
        northness = np.cos(np.radians(aspect_deg))
        eastness = np.sin(np.radians(aspect_deg))
        return np.column_stack(
            [elevation, slope_deg, aspect_deg, twi, curvature, northness, eastness]
        )

    # Fall back to on-the-fly computation from dem.tif
    return _compute_from_dem(dem_path, points)
