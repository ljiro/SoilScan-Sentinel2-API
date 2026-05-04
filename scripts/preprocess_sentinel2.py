"""
One-time preprocessing script: convert raw .SAFE tiles into two compact GeoTIFFs.

Run this locally where your .SAFE tiles are stored, then upload only the two
output files (~50-150 MB total) to the Railway Volume instead of GB-scale .SAFE dirs.

Output files (written to --out-dir):
    bands_mean.tif  — 12-band GeoTIFF (mean across tiles, 10 m resolution, AOI-clipped)
    bands_std.tif   — 12-band GeoTIFF (temporal std across tiles; all NaN if single tile)

Band order (same as training pipeline):
    1=B01  2=B02  3=B03  4=B04  5=B05  6=B06
    7=B07  8=B08  9=B8A  10=B09 11=B11 12=B12

Usage:
    python scripts/preprocess_sentinel2.py \
        --safe-dir data/sentinel2 \
        --out-dir  data/sentinel2 \
        --aoi 120.3 16.2 120.85 16.85

    # AOI format: minlon minlat maxlon maxlat (WGS84)
    # Default AOI covers Benguet province, Philippines
"""
import argparse
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject

BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
_BAND_RE = re.compile(r"_(B(?:0[1-9]|1[0-2]|8A))_")
_RES_DIRS = ["R10m", "R20m", "R60m"]

TARGET_RES_M = 10
TARGET_CRS = CRS.from_epsg(32651)  # UTM Zone 51N — Philippines

# Default AOI: Benguet province, Philippines
DEFAULT_AOI = (120.3, 16.2, 120.85, 16.85)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _compute_target_grid(aoi_wgs84: Tuple[float, float, float, float]):
    """Return (transform, width, height) for the target UTM grid."""
    lon_min, lat_min, lon_max, lat_max = aoi_wgs84
    t = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    x_min, y_min = t.transform(lon_min, lat_min)
    x_max, y_max = t.transform(lon_max, lat_max)

    # Snap extents to TARGET_RES grid
    x_min = np.floor(x_min / TARGET_RES_M) * TARGET_RES_M
    y_min = np.floor(y_min / TARGET_RES_M) * TARGET_RES_M
    x_max = np.ceil(x_max  / TARGET_RES_M) * TARGET_RES_M
    y_max = np.ceil(y_max  / TARGET_RES_M) * TARGET_RES_M

    width  = int(round((x_max - x_min) / TARGET_RES_M))
    height = int(round((y_max - y_min) / TARGET_RES_M))
    transform = rasterio.transform.from_origin(x_min, y_max, TARGET_RES_M, TARGET_RES_M)
    return transform, width, height


def _reproject_band(
    src_path: Path,
    dst_transform,
    dst_width: int,
    dst_height: int,
) -> Optional[np.ndarray]:
    """Reproject a single band JP2 to the target grid. Returns (H, W) float32 or None."""
    try:
        with rasterio.open(src_path) as src:
            dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
            return dst
    except Exception as e:
        print(f"    [warn] could not reproject {src_path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _open_output(path: Path, transform, width: int, height: int) -> rasterio.DatasetWriter:
    return rasterio.open(
        path, "w",
        driver="GTiff",
        height=height, width=width,
        count=len(BAND_NAMES),
        dtype="float32",
        crs=TARGET_CRS,
        transform=transform,
        compress="deflate",
        predictor=2,
        nodata=np.nan,
    )


def preprocess(safe_dir: Path, out_dir: Path, aoi: Tuple[float, float, float, float]):
    safe_dirs = sorted(safe_dir.glob("*.SAFE"))
    if not safe_dirs:
        raise FileNotFoundError(f"No .SAFE directories found in {safe_dir}")

    print(f"Found {len(safe_dirs)} .SAFE tile(s)")
    transform, width, height = _compute_target_grid(aoi)
    print(f"Target grid: {width} × {height} px at {TARGET_RES_M} m  ({TARGET_CRS.to_epsg()})")
    print(f"AOI (WGS84): lon {aoi[0]}–{aoi[2]}  lat {aoi[1]}–{aoi[3]}")

    # Collect band file maps per tile upfront
    all_band_files = []
    for tile_path in safe_dirs:
        bf = _find_band_files(tile_path)
        if bf:
            all_band_files.append((tile_path.name, bf))
    n_tiles = len(all_band_files)

    out_dir.mkdir(parents=True, exist_ok=True)
    mean_path = out_dir / "bands_mean.tif"
    std_path  = out_dir / "bands_std.tif"

    # Process one band at a time to keep peak RAM at ~1 band × n_tiles
    with _open_output(mean_path, transform, width, height) as mean_ds, \
         _open_output(std_path,  transform, width, height) as std_ds:

        for j, band in enumerate(BAND_NAMES, start=1):
            print(f"\n[{j:02d}/12] {band}")
            band_arrays: List[np.ndarray] = []

            for tile_name, band_files in all_band_files:
                print(f"  {tile_name[:40]} ...", end=" ", flush=True)
                if band not in band_files:
                    print("not found — NaN")
                    band_arrays.append(np.full((height, width), np.nan, dtype=np.float32))
                    continue
                arr = _reproject_band(band_files[band], transform, width, height)
                if arr is None:
                    arr = np.full((height, width), np.nan, dtype=np.float32)
                valid = int(np.sum(~np.isnan(arr)))
                print(f"{valid:,} valid px")
                band_arrays.append(arr)

            stacked = np.stack(band_arrays, axis=0)  # (T, H, W)
            mean_ds.write(np.nanmean(stacked, axis=0).astype(np.float32), j)
            std_ds.write(
                np.nanstd(stacked, axis=0).astype(np.float32) if n_tiles > 1
                else np.full((height, width), np.nan, dtype=np.float32),
                j,
            )

    print(f"\nDone.")
    print(f"  {mean_path}  ({mean_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {std_path}   ({std_path.stat().st_size  / 1e6:.1f} MB)")
    print("\nUpload both files to the Railway Volume at:")
    print("  /mnt/soilscan-data/sentinel2/bands_mean.tif")
    print("  /mnt/soilscan-data/sentinel2/bands_std.tif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess .SAFE tiles → compact GeoTIFFs")
    parser.add_argument("--safe-dir", type=Path,
                        default=Path("D:/Github/SoilScan-Sentinel2/data/raw/field_products"),
                        help="Directory containing .SAFE folders")
    parser.add_argument("--out-dir",  type=Path,
                        default=Path("data/sentinel2"),
                        help="Output directory for bands_mean.tif and bands_std.tif")
    parser.add_argument("--aoi", type=float, nargs=4,
                        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
                        default=DEFAULT_AOI,
                        help="Bounding box in WGS84 (default: Benguet province)")
    args = parser.parse_args()
    preprocess(args.safe_dir, args.out_dir, tuple(args.aoi))
