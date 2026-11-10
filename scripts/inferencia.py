"""Inferencia del modelo ganador de House Pricing.

Carga el pipeline de inferencia generado en el notebook
``2.0-rf-modelado`` (selección de features, ingeniería de
características y GradientBoosting) y predice ``SalePrice`` en dólares
a partir de datos crudos con el formato de ``test.csv``.

Cada ejecución queda registrada en ``scripts/inferencia.log``, con la
fecha y hora en cada línea. El log incluye un control de calidad entre
la entrada y la salida: filas recibidas, filas con predicción coherente
(finita y dentro del rango de precios de train) y filas no coherentes.

Está pensado para consumirse desde una API (por ejemplo, FastAPI) con
``predecir`` o ``predecir_registros``, o desde la línea de comandos::

    python scripts/inferencia.py --entrada data/raw/test.csv \\
        --salida data/processed/predicciones_test.csv
"""

import argparse
from functools import lru_cache
from pathlib import Path
import time
from typing import Any

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
import yaml

from house_pricing.config import MODELS_DIR, PROJ_ROOT

RUTA_PIPELINE = MODELS_DIR / "pipeline_inferencia.joblib"
RUTA_CONFIGURACION = PROJ_ROOT / "pipeline_config.yaml"
RUTA_LOG = Path(__file__).resolve().parent / "inferencia.log"

COLUMNA_ID = "Id"
COLUMNA_PREDICCION = "SalePrice"
COLUMNA_COHERENCIA = "es_coherente"

FORMATO_LOG = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{function}:{line} | {message}"
)

_id_handler_archivo: int | None = None


class ErrorEntradaInvalida(ValueError):
    """Los datos de entrada no tienen el formato necesario."""


class ErrorPrediccion(RuntimeError):
    """El pipeline falló o no generó ninguna predicción coherente."""


def configurar_logging(ruta_log: Path = RUTA_LOG) -> None:
    """Agrega el archivo de log, una sola vez por proceso.

    Args:
        ruta_log: Archivo donde se agregan los mensajes de log.
    """
    global _id_handler_archivo
    if _id_handler_archivo is not None:
        return
    _id_handler_archivo = logger.add(
        ruta_log,
        level="INFO",
        format=FORMATO_LOG,
        encoding="utf-8",
        buffering=1,
    )


@lru_cache(maxsize=1)
def cargar_configuracion(
    ruta: Path = RUTA_CONFIGURACION,
) -> dict[str, Any]:
    """Lee la configuración del pipeline.

    Args:
        ruta: Ruta de ``pipeline_config.yaml``.

    Returns:
        Diccionario con ``MODEL_FEATURES`` y ``SALEPRICE_RANGE``.

    Raises:
        FileNotFoundError: Si el archivo de configuración no existe.
        KeyError: Si falta alguna de las claves necesarias.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la configuración: {ruta}")
    with open(ruta, encoding="utf-8") as archivo:
        configuracion = yaml.safe_load(archivo)
    for clave in ("MODEL_FEATURES", "SALEPRICE_RANGE"):
        if clave not in configuracion:
            raise KeyError(f"Falta la clave {clave} en {ruta}")
    return configuracion


@lru_cache(maxsize=1)
def cargar_pipeline(
    ruta: Path = RUTA_PIPELINE,
) -> TransformedTargetRegressor:
    """Carga el pipeline de inferencia entrenado.

    El resultado queda en caché para no leer el archivo en cada
    petición.

    Args:
        ruta: Ruta de ``pipeline_inferencia.joblib``.

    Returns:
        Pipeline completo: selección de features, ingeniería de
        características y modelo, con la salida en dólares.

    Raises:
        FileNotFoundError: Si el archivo del pipeline no existe.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el pipeline: {ruta}")
    pipeline = joblib.load(ruta)
    logger.info(f"Pipeline cargado desde {ruta}")
    return pipeline


def obtener_features_numericas(
    pipeline: TransformedTargetRegressor,
) -> list[str]:
    """Obtiene las features que el pipeline trata como numéricas.

    Args:
        pipeline: Pipeline de inferencia ya entrenado.

    Returns:
        Nombres de las columnas numéricas que esperan los imputadores.
    """
    pasos = pipeline.regressor_.named_steps["features"].named_steps
    return (
        list(pasos["imputar_aleatorio"].variables_)
        + list(pasos["imputar_mediana"].variables_)
    )


def _error_entrada(mensaje: str) -> ErrorEntradaInvalida:
    """Registra el mensaje en el log y crea la excepción.

    Args:
        mensaje: Descripción del problema con la entrada.

    Returns:
        Excepción lista para lanzar.
    """
    logger.error(mensaje)
    return ErrorEntradaInvalida(mensaje)


def validar_entrada(
    datos: pd.DataFrame,
    features: list[str],
    features_numericas: list[str],
) -> pd.DataFrame:
    """Valida los datos crudos antes de predecir.

    Los valores faltantes están permitidos porque el pipeline los
    imputa, pero se registran como advertencia.

    Args:
        datos: Datos crudos, una casa por fila.
        features: Columnas que necesita el modelo.
        features_numericas: Columnas que deben contener números.

    Returns:
        Copia de ``datos`` con las columnas numéricas convertidas a
        número (por ejemplo, ``"1500"`` pasa a ``1500``) y las
        categóricas vacías convertidas a texto.

    Raises:
        ErrorEntradaInvalida: Si la entrada no es un DataFrame, está
            vacía, le faltan columnas o tiene texto en columnas
            numéricas.
    """
    if not isinstance(datos, pd.DataFrame):
        raise _error_entrada(
            f"Se esperaba un DataFrame y se recibió {type(datos).__name__}"
        )
    if datos.empty:
        raise _error_entrada("La entrada no tiene filas")

    faltantes = [col for col in features if col not in datos.columns]
    if faltantes:
        raise _error_entrada(f"Faltan columnas en la entrada: {faltantes}")

    datos = datos.copy()
    for col in features_numericas:
        convertida = pd.to_numeric(datos[col], errors="coerce")
        invalidos = convertida.isna() & datos[col].notna()
        if invalidos.any():
            ids = datos.index[invalidos].tolist()
            raise _error_entrada(
                f"La columna {col} tiene valores no numéricos en "
                f"{COLUMNA_ID}={ids}"
            )
        datos[col] = convertida

    # Una columna categórica sin ningún valor (por ejemplo, una sola casa
    # sin garaje) llega como numérica. Se pasa al tipo texto de pandas,
    # el mismo que tenía en entrenamiento, para poder imputarla con la
    # moda. Con ``object`` no basta: los imputadores de feature-engine
    # llaman a ``infer_objects()`` y la regresan a numérica.
    for col in features:
        if col not in features_numericas and (
            pd.api.types.is_numeric_dtype(datos[col])
        ):
            datos[col] = datos[col].astype("str")

    if datos.index.has_duplicates:
        logger.warning(
            f"Hay {datos.index.duplicated().sum()} valores de "
            f"{COLUMNA_ID} duplicados"
        )
    filas_con_faltantes = int(datos[features].isna().any(axis=1).sum())
    if filas_con_faltantes:
        logger.warning(
            f"{filas_con_faltantes} filas tienen valores faltantes; "
            "el pipeline los imputará"
        )
    return datos


def evaluar_coherencia(
    predicciones: pd.Series,
    minimo: float,
    maximo: float,
) -> pd.Series:
    """Marca qué predicciones son coherentes.

    Una predicción es coherente si es finita y está dentro del rango
    de precios observado en train.

    Args:
        predicciones: Precios predichos en dólares.
        minimo: Precio mínimo aceptado.
        maximo: Precio máximo aceptado.

    Returns:
        Serie booleana con el mismo índice que ``predicciones``.
    """
    finitas = np.isfinite(predicciones)
    en_rango = predicciones.between(minimo, maximo)
    return (finitas & en_rango).rename(COLUMNA_COHERENCIA)


def predecir(datos: pd.DataFrame) -> pd.DataFrame:
    """Predice ``SalePrice`` en dólares a partir de datos crudos.

    Args:
        datos: Datos crudos con el formato de ``test.csv``. Puede
            tener columnas de más; solo se usan las de
            ``MODEL_FEATURES``. El índice se usa como ``Id`` en el log
            y en la salida.

    Returns:
        DataFrame con el mismo índice que ``datos`` y las columnas
        ``SalePrice`` (precio predicho) y ``es_coherente``.

    Raises:
        ErrorEntradaInvalida: Si los datos no pasan la validación.
        ErrorPrediccion: Si el pipeline falla, si la salida no tiene
            una predicción por fila o si ninguna es coherente.
    """
    configurar_logging()
    inicio = time.perf_counter()
    filas_entrada = len(datos) if isinstance(datos, pd.DataFrame) else 0
    logger.info(f"Inicio de inferencia: {filas_entrada} filas recibidas")

    configuracion = cargar_configuracion()
    pipeline = cargar_pipeline()
    datos_validos = validar_entrada(
        datos,
        configuracion["MODEL_FEATURES"],
        obtener_features_numericas(pipeline),
    )

    try:
        valores = pipeline.predict(datos_validos)
    except Exception as error:
        logger.exception("El pipeline falló al predecir")
        raise ErrorPrediccion("El pipeline falló al predecir") from error

    if len(valores) != filas_entrada:
        mensaje = (
            f"El pipeline devolvió {len(valores)} predicciones para "
            f"{filas_entrada} filas"
        )
        logger.error(mensaje)
        raise ErrorPrediccion(mensaje)

    predicciones = pd.Series(
        valores, index=datos_validos.index, name=COLUMNA_PREDICCION
    )
    rango = configuracion["SALEPRICE_RANGE"]
    coherentes = evaluar_coherencia(predicciones, rango["min"], rango["max"])
    for id_fila, valor in predicciones[~coherentes].items():
        logger.error(
            f"Predicción no coherente para {COLUMNA_ID}={id_fila}: "
            f"{valor} (rango aceptado: {rango['min']}-{rango['max']})"
        )

    filas_coherentes = int(coherentes.sum())
    logger.info(
        f"Control de calidad | filas de entrada: {filas_entrada} | "
        f"predicciones: {len(predicciones)} | "
        f"coherentes: {filas_coherentes} | "
        f"no coherentes: {filas_entrada - filas_coherentes} | "
        f"tiempo: {time.perf_counter() - inicio:.3f} s"
    )
    if filas_coherentes == 0:
        mensaje = "Ninguna predicción es coherente"
        logger.error(mensaje)
        raise ErrorPrediccion(mensaje)

    return pd.concat([predicciones, coherentes], axis=1)


def predecir_registros(
    registros: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Predice a partir de registros JSON, como los que recibe una API.

    Args:
        registros: Lista de casas, cada una como un diccionario
            ``{columna: valor}``. Si incluyen la clave ``Id``, se usa
            para identificar cada casa.

    Returns:
        Lista de diccionarios con ``Id``, ``SalePrice`` y
        ``es_coherente``, en el mismo orden de la entrada.

    Raises:
        ErrorEntradaInvalida: Si la lista está vacía o los datos no
            pasan la validación.
        ErrorPrediccion: Si la predicción falla.
    """
    if not registros:
        configurar_logging()
        raise _error_entrada("La lista de registros está vacía")

    datos = pd.DataFrame.from_records(registros)
    if COLUMNA_ID in datos.columns:
        datos = datos.set_index(COLUMNA_ID)
    resultado = predecir(datos)
    return (
        resultado.rename_axis(COLUMNA_ID)
        .reset_index()
        .to_dict(orient="records")
    )


def main() -> None:
    """Ejecuta la inferencia sobre un CSV desde la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Predice SalePrice a partir de un CSV crudo."
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        required=True,
        help="CSV con el formato de test.csv (con columna Id).",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=None,
        help="CSV donde guardar Id y SalePrice (opcional).",
    )
    argumentos = parser.parse_args()

    datos = pd.read_csv(argumentos.entrada, index_col=COLUMNA_ID)
    resultado = predecir(datos)

    if argumentos.salida is not None:
        resultado[[COLUMNA_PREDICCION]].to_csv(
            argumentos.salida, index_label=COLUMNA_ID
        )
        print(f"Predicciones guardadas en {argumentos.salida}")
    print(
        f"Filas: {len(resultado)} | "
        f"coherentes: {int(resultado[COLUMNA_COHERENCIA].sum())} | "
        f"log: {RUTA_LOG}"
    )


if __name__ == "__main__":
    main()
