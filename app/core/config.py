from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    sentinel2_dir: Path = Path("data/sentinel2")
    soilgrids_dir: Path = Path("data/soilgrids")
    dem_path: Path = Path("data/dem/dem.tif")
    models_dir: Path = Path("models")

    # Grid sampling defaults
    max_sample_points: int = 500
    default_sample_spacing_m: float = 10.0

    # Microclimate defaults (Benguet highlands medians)
    default_temperature_c: float = 18.0
    default_humidity_percent: float = 80.0

    # Admin token for the /admin/download endpoint (set via SOILSCAN_ADMIN_TOKEN env var)
    admin_token: str = ""

    class Config:
        env_prefix = "SOILSCAN_"


settings = Settings()
