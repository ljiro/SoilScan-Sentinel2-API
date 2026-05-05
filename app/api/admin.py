"""
Admin endpoints for uploading data files into the Railway Volume.

POST /admin/download
  Accepts a Google Drive or direct URL and downloads the file into the Volume.
  Protected by the SOILSCAN_ADMIN_TOKEN environment variable.

Predefined targets map friendly names to the correct Volume paths so callers
don't need to know the internal directory layout.
"""
import re
import urllib.request
from pathlib import Path
from typing import Optional

import gdown
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter(prefix="/admin", tags=["Admin"])

# Friendly name → path relative to the volume root
_TARGETS = {
    "bands_mean":            lambda: settings.sentinel2_dir / "bands_mean.tif",
    "bands_std":             lambda: settings.sentinel2_dir / "bands_std.tif",
    "dem":                   lambda: settings.dem_path,
    "sg_phh2o_0-5cm":        lambda: settings.soilgrids_dir / "phh2o" / "phh2o_0-5cm_mean.tif",
    "sg_phh2o_5-15cm":       lambda: settings.soilgrids_dir / "phh2o" / "phh2o_5-15cm_mean.tif",
    "sg_soc_0-5cm":          lambda: settings.soilgrids_dir / "soc"   / "soc_0-5cm_mean.tif",
    "sg_soc_5-15cm":         lambda: settings.soilgrids_dir / "soc"   / "soc_5-15cm_mean.tif",
    "sg_nitrogen_0-5cm":     lambda: settings.soilgrids_dir / "nitrogen" / "nitrogen_0-5cm_mean.tif",
    "sg_nitrogen_5-15cm":    lambda: settings.soilgrids_dir / "nitrogen" / "nitrogen_5-15cm_mean.tif",
    "sg_clay_0-5cm":         lambda: settings.soilgrids_dir / "clay"  / "clay_0-5cm_mean.tif",
    "sg_clay_5-15cm":        lambda: settings.soilgrids_dir / "clay"  / "clay_5-15cm_mean.tif",
    "sg_sand_0-5cm":         lambda: settings.soilgrids_dir / "sand"  / "sand_0-5cm_mean.tif",
    "sg_sand_5-15cm":        lambda: settings.soilgrids_dir / "sand"  / "sand_5-15cm_mean.tif",
    "sg_cec_0-5cm":          lambda: settings.soilgrids_dir / "cec"   / "cec_0-5cm_mean.tif",
    "sg_cec_5-15cm":         lambda: settings.soilgrids_dir / "cec"   / "cec_5-15cm_mean.tif",
}

_GDRIVE_RE = re.compile(r"drive\.google\.com/file/d/([^/]+)")


def _resolve_gdrive_url(url: str) -> str:
    """Convert a Google Drive share link to a direct download URL."""
    m = _GDRIVE_RE.search(url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}&confirm=t"
    return url


def _check_token(token: Optional[str]):
    if not settings.admin_token:
        raise HTTPException(status_code=503, detail="SOILSCAN_ADMIN_TOKEN is not set on the server.")
    if token != settings.admin_token:
        raise HTTPException(status_code=401, detail="Invalid admin token.")


class DownloadRequest(BaseModel):
    url: str
    target: str


@router.post("/download")
def download_file(
    req: DownloadRequest,
    x_admin_token: Optional[str] = Header(default=None),
):
    """
    Download a file from a URL (including Google Drive share links) into the Volume.

    target must be one of the predefined keys — call GET /admin/targets to list them.
    Requires the X-Admin-Token header to match SOILSCAN_ADMIN_TOKEN env var.
    """
    _check_token(x_admin_token)

    if req.target not in _TARGETS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown target '{req.target}'. Valid targets: {list(_TARGETS.keys())}",
        )

    dest: Path = _TARGETS[req.target]()
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        if _GDRIVE_RE.search(req.url):
            # gdown handles Google Drive's large-file confirmation page automatically
            gdown.download(req.url, str(dest), quiet=False)
        else:
            urllib.request.urlretrieve(req.url, dest)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Download failed: {exc}")

    size_mb = dest.stat().st_size / 1e6
    return {
        "status": "ok",
        "target": req.target,
        "dest": str(dest),
        "size_mb": round(size_mb, 2),
    }


@router.get("/targets")
def list_targets(x_admin_token: Optional[str] = Header(default=None)):
    """List all valid download target names and their resolved paths."""
    _check_token(x_admin_token)
    return {k: str(v()) for k, v in _TARGETS.items()}


@router.get("/files")
def list_files(x_admin_token: Optional[str] = Header(default=None)):
    """Show which Volume files are present and their sizes."""
    _check_token(x_admin_token)
    result = {}
    for name, path_fn in _TARGETS.items():
        p = path_fn()
        result[name] = {
            "path": str(p),
            "exists": p.exists(),
            "size_mb": round(p.stat().st_size / 1e6, 2) if p.exists() else None,
        }
    return result
