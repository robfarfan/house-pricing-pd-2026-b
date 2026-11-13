import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# MLflow: tracking y Model Registry en SQLite local, artefactos en mlruns/.
# Se pueden redirigir con variables de entorno (por ejemplo, en .env).
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
)
MLFLOW_ARTIFACTS_URI = os.getenv(
    "MLFLOW_ARTIFACTS_URI", (PROJ_ROOT / "mlruns").as_uri()
)
MLFLOW_EXPERIMENT_ENTRENAMIENTO = "house_pricing-entrenamiento"
MLFLOW_EXPERIMENT_INFERENCIA = "house_pricing-inferencia"
MLFLOW_MODEL_NAME = "house_pricing"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
