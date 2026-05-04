"""
Compute spectral indices from Sentinel-2 band arrays.

Replicates _add_spectral_indices() from SoilScan-Sentinel2/src/train_ordinal.py exactly.
All formulas use _EPS = 1e-6 to avoid division by zero.

Input:
    bands — (N, 12) array in BAND_NAMES order:
            B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12

Output:
    (N, 10) array in INDEX_NAMES order:
            NDVI EVI SAVI MSAVI NDRE CHL_re BSI BI NDWI NDMI
"""
import numpy as np

# Must match sentinel2_extractor.BAND_NAMES exactly
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
_B = {name: i for i, name in enumerate(BAND_NAMES)}

INDEX_NAMES = ["NDVI", "EVI", "SAVI", "MSAVI", "NDRE", "CHL_re", "BSI", "BI", "NDWI", "NDMI"]

_EPS = 1e-6


def compute_indices(bands: np.ndarray) -> np.ndarray:
    """
    Args:
        bands: (N, 12) float array of reflectance values in BAND_NAMES order.
    Returns:
        (N, 10) float array of spectral index values in INDEX_NAMES order.
    """
    b02 = bands[:, _B["B02"]].astype(float)
    b03 = bands[:, _B["B03"]].astype(float)
    b04 = bands[:, _B["B04"]].astype(float)
    b05 = bands[:, _B["B05"]].astype(float)
    b08 = bands[:, _B["B08"]].astype(float)
    b8a = bands[:, _B["B8A"]].astype(float)
    b11 = bands[:, _B["B11"]].astype(float)

    ndvi  = (b08 - b04) / (b08 + b04 + _EPS)
    evi   = 2.5 * (b08 - b04) / (b08 + 6 * b04 - 7.5 * b02 + 1 + _EPS)
    savi  = 1.5 * (b08 - b04) / (b08 + b04 + 0.5 + _EPS)
    msavi = (2 * b08 + 1 - np.sqrt(np.maximum((2 * b08 + 1) ** 2 - 8 * (b08 - b04), 0))) / 2
    ndre  = (b8a - b05) / (b8a + b05 + _EPS)
    chl_re = (b8a / (b05 + _EPS)) - 1
    bsi   = ((b11 + b04) - (b08 + b02)) / ((b11 + b04) + (b08 + b02) + _EPS)
    bi    = np.sqrt((b04 ** 2 + b08 ** 2) / 2)
    ndwi  = (b03 - b08) / (b03 + b08 + _EPS)
    ndmi  = (b08 - b11) / (b08 + b11 + _EPS)

    return np.column_stack([ndvi, evi, savi, msavi, ndre, chl_re, bsi, bi, ndwi, ndmi])
