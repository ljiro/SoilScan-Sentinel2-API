from typing import Any, Dict, List, Tuple

import numpy as np
import pyproj
from shapely.geometry import Point, shape
from shapely.ops import transform as shp_transform


def _utm_crs(lon: float, lat: float) -> pyproj.CRS:
    zone = int((lon + 180) / 6) + 1
    return pyproj.CRS.from_dict(
        {"proj": "utm", "zone": zone, "datum": "WGS84", "south": lat < 0}
    )


def sample_polygon(
    geojson: Dict[str, Any],
    spacing_m: float = 10.0,
    max_points: int = 500,
) -> List[Tuple[float, float]]:
    """
    Generate a regular grid of (lon, lat) points inside a GeoJSON polygon.

    Returns at most `max_points` points; if the grid exceeds the cap, points
    are drawn uniformly at random from the full grid.
    """
    polygon = shape(geojson)
    centroid = polygon.centroid
    utm = _utm_crs(centroid.x, centroid.y)

    to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform
    to_wgs = pyproj.Transformer.from_crs(utm, "EPSG:4326", always_xy=True).transform

    poly_utm = shp_transform(to_utm, polygon)
    minx, miny, maxx, maxy = poly_utm.bounds

    xs = np.arange(minx, maxx, spacing_m)
    ys = np.arange(miny, maxy, spacing_m)

    points: List[Tuple[float, float]] = []
    for x in xs:
        for y in ys:
            if poly_utm.contains(Point(x, y)):
                lon, lat = to_wgs(x, y)
                points.append((float(lon), float(lat)))

    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = [points[i] for i in idx]

    return points


def polygon_area_ha(geojson: Dict[str, Any]) -> float:
    """Return polygon area in hectares using an equal-area projection."""
    polygon = shape(geojson)
    centroid = polygon.centroid
    utm = _utm_crs(centroid.x, centroid.y)
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform
    poly_utm = shp_transform(to_utm, polygon)
    return poly_utm.area / 10_000
