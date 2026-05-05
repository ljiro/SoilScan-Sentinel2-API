"""
Admin endpoints for uploading data files into the Railway Volume.

POST /admin/download
  Accepts a Google Drive or direct URL and downloads the file into the Volume.
  Protected by the SOILSCAN_ADMIN_TOKEN environment variable.

Predefined targets map friendly names to the correct Volume paths so callers
don't need to know the internal directory layout.
"""
import os
import re
import tempfile
import urllib.request
import zipfile
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


@router.get("/ls")
def list_dir(
    path: str = "/mnt/soilscan-data",
    x_admin_token: Optional[str] = Header(default=None),
):
    """List files under a directory path (default: /mnt/soilscan-data)."""
    _check_token(x_admin_token)
    result = {}
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            result[str(fp)] = round(fp.stat().st_size / 1e6, 3)
    return {"base": path, "files": result}


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


class UnzipRequest(BaseModel):
    url: str
    dest_dir: str  # one of: "soilgrids", "sentinel2", "dem"


_UNZIP_DIRS = {
    "soilgrids":  lambda: settings.soilgrids_dir,
    "sentinel2":  lambda: settings.sentinel2_dir,
    "dem":        lambda: settings.dem_path.parent,
}


@router.post("/unzip")
def unzip_file(
    req: UnzipRequest,
    x_admin_token: Optional[str] = Header(default=None),
):
    """
    Download a zip from a URL (including Google Drive) and extract it into a Volume directory.

    dest_dir must be one of: soilgrids, sentinel2, dem.
    Requires the X-Admin-Token header.
    """
    _check_token(x_admin_token)

    if req.dest_dir not in _UNZIP_DIRS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown dest_dir '{req.dest_dir}'. Valid options: {list(_UNZIP_DIRS.keys())}",
        )

    dest_dir: Path = _UNZIP_DIRS[req.dest_dir]()
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        if _GDRIVE_RE.search(req.url):
            gdown.download(req.url, str(tmp_path), quiet=False)
        else:
            urllib.request.urlretrieve(req.url, tmp_path)

        with zipfile.ZipFile(tmp_path, "r") as zf:
            extracted = []
            for member in zf.infolist():
                # Normalize Windows backslash paths so they extract correctly on Linux
                norm_name = member.filename.replace("\\", "/")
                dest_file = dest_dir / norm_name
                if member.is_dir() or norm_name.endswith("/"):
                    dest_file.mkdir(parents=True, exist_ok=True)
                else:
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(dest_file, "wb") as dst:
                        dst.write(src.read())
                    extracted.append(norm_name)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=502, detail="Downloaded file is not a valid zip.")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Unzip failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "status": "ok",
        "dest_dir": str(dest_dir),
        "files_extracted": len(extracted),
        "files": extracted,
    }
