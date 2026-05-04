"""
One-time script: pre-compute terrain attribute rasters from a DEM GeoTIFF.

Generates slope.tif, aspect.tif, twi.tif, curvature.tif alongside dem.tif.
These are loaded directly by terrain_extractor.py at inference time, which is
faster and more accurate than on-the-fly computation per request.

Requires richdem:
    pip install richdem

Usage:
    python scripts/preprocess_terrain.py --dem data/dem/dem.tif
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

try:
    import richdem as rd
    _HAS_RICHDEM = True
except ImportError:
    _HAS_RICHDEM = False


def _write_raster(path: Path, data: np.ndarray, ref_ds: rasterio.DatasetReader):
    profile = ref_ds.profile.copy()
    profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)
    print(f"  written: {path}")


def preprocess(dem_path: Path):
    out_dir = dem_path.parent
    print(f"Reading DEM: {dem_path}")

    with rasterio.open(dem_path) as src:
        dem_np = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            dem_np[dem_np == nodata] = np.nan
        profile = src.profile

    if _HAS_RICHDEM:
        dem_rd = rd.rdarray(dem_np, no_data=np.nan)
        dem_rd.geotransform = [
            profile["transform"].c,
            profile["transform"].a,
            0,
            profile["transform"].f,
            0,
            profile["transform"].e,
        ]

        slope = np.array(rd.TerrainAttribute(dem_rd, attrib="slope_degrees"))
        aspect = np.array(rd.TerrainAttribute(dem_rd, attrib="aspect"))
        curvature = np.array(rd.TerrainAttribute(dem_rd, attrib="curvature"))

        # TWI: ln(upslope_area / tan(slope))
        accum = np.array(rd.FlowAccumulation(dem_rd, method="D8")) * (
            abs(profile["transform"].a) * abs(profile["transform"].e)
        )
        slope_rad = np.radians(np.clip(slope, 0.001, None))
        twi = np.log((accum + 1e-6) / np.tan(slope_rad))
    else:
        print("richdem not found — computing slope/aspect with numpy gradients (approximate).")
        mid_lat = profile["transform"].f + profile["transform"].e * dem_np.shape[0] / 2
        cell_x_m = abs(profile["transform"].a) * 111_320 * np.cos(np.radians(mid_lat))
        cell_y_m = abs(profile["transform"].e) * 111_320

        dy, dx = np.gradient(dem_np, cell_y_m, cell_x_m)
        slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
        slope = np.degrees(slope_rad)
        aspect = np.degrees(np.arctan2(-dx, dy)) % 360

        d2x = np.gradient(dx, cell_x_m, axis=1)
        d2y = np.gradient(dy, cell_y_m, axis=0)
        curvature = -(d2x + d2y)

        twi = np.log(1.0 / np.tan(np.clip(slope_rad, 1e-4, None)))

    with rasterio.open(dem_path) as src:
        _write_raster(out_dir / "slope.tif", slope, src)
        _write_raster(out_dir / "aspect.tif", aspect, src)
        _write_raster(out_dir / "curvature.tif", curvature, src)
        _write_raster(out_dir / "twi.tif", twi, src)

    print("Done. Terrain rasters written to", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", default="data/dem/dem.tif", type=Path)
    args = parser.parse_args()
    preprocess(args.dem)
