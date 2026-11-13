"""Configuración de MLflow compartida por los notebooks del proyecto."""

import mlflow
from loguru import logger

from house_pricing.config import MLFLOW_ARTIFACTS_URI, MLFLOW_TRACKING_URI


def configurar_mlflow(experimento: str) -> str:
    """Apunta MLflow al backend del proyecto y activa un experimento.

    Si el experimento no existe, lo crea con sus artefactos en
    ``mlruns/<experimento>``. Se fija esa ruta absoluta porque la ruta
    por defecto de MLflow es relativa al directorio de trabajo, y los
    notebooks se ejecutan desde ``notebooks/``.

    Args:
        experimento: Nombre del experimento en MLflow.

    Returns:
        El id del experimento activo.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(experimento) is None:
        mlflow.create_experiment(
            experimento,
            artifact_location=f"{MLFLOW_ARTIFACTS_URI}/{experimento}",
        )
        logger.info(f"Experimento de MLflow creado: {experimento}")
    experimento_id = mlflow.set_experiment(experimento).experiment_id
    logger.info(
        f"MLflow: {MLFLOW_TRACKING_URI} | experimento {experimento} "
        f"(id {experimento_id})"
    )
    return experimento_id
