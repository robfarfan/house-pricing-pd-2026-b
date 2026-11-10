"""Pruebas de ``scripts/inferencia.py``.

Ejecutar desde la carpeta ``House_Pricing``::

    pytest
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from house_pricing.config import RAW_DATA_DIR
import inferencia
from inferencia import (
    COLUMNA_COHERENCIA,
    COLUMNA_ID,
    COLUMNA_PREDICCION,
    ErrorEntradaInvalida,
    ErrorPrediccion,
    RUTA_LOG,
    evaluar_coherencia,
    predecir,
    predecir_registros,
)


@pytest.fixture(scope="module")
def test_crudo() -> pd.DataFrame:
    """Set de test crudo de Kaggle, con ``Id`` como índice."""
    return pd.read_csv(RAW_DATA_DIR / "test.csv", index_col=COLUMNA_ID)


def _pipeline_falso(predecir_falso) -> SimpleNamespace:
    """Envuelve el pipeline real reemplazando su método ``predict``."""
    real = inferencia.cargar_pipeline()
    return SimpleNamespace(regressor_=real.regressor_, predict=predecir_falso)


@pytest.mark.parametrize("semilla", [0, 1, 2, 3, 4])
def test_muestra_aleatoria(test_crudo: pd.DataFrame, semilla: int) -> None:
    muestra = test_crudo.sample(n=25, random_state=semilla)

    resultado = predecir(muestra)

    assert len(resultado) == len(muestra)
    assert resultado.index.equals(muestra.index)
    assert np.isfinite(resultado[COLUMNA_PREDICCION]).all()
    assert resultado[COLUMNA_COHERENCIA].all()


def test_test_completo(test_crudo: pd.DataFrame) -> None:
    resultado = predecir(test_crudo)

    assert len(resultado) == len(test_crudo)
    assert resultado[COLUMNA_COHERENCIA].all()


def test_una_sola_fila(test_crudo: pd.DataFrame) -> None:
    resultado = predecir(test_crudo.sample(n=1, random_state=7))

    assert len(resultado) == 1
    assert resultado[COLUMNA_COHERENCIA].iloc[0]


def test_columna_faltante_lanza_error(test_crudo: pd.DataFrame) -> None:
    muestra = test_crudo.sample(n=5, random_state=0).drop(
        columns="GrLivArea"
    )

    with pytest.raises(ErrorEntradaInvalida, match="GrLivArea"):
        predecir(muestra)


def test_entrada_vacia_lanza_error(test_crudo: pd.DataFrame) -> None:
    with pytest.raises(ErrorEntradaInvalida, match="no tiene filas"):
        predecir(test_crudo.iloc[0:0])


def test_entrada_que_no_es_dataframe_lanza_error() -> None:
    with pytest.raises(ErrorEntradaInvalida, match="DataFrame"):
        predecir([1, 2, 3])


def test_texto_en_columna_numerica_lanza_error(
    test_crudo: pd.DataFrame,
) -> None:
    muestra = test_crudo.sample(n=5, random_state=0)
    muestra["GrLivArea"] = muestra["GrLivArea"].astype(object)
    muestra.iloc[0, muestra.columns.get_loc("GrLivArea")] = "abc"

    with pytest.raises(ErrorEntradaInvalida, match="no numéricos"):
        predecir(muestra)


def test_numeros_como_texto_se_convierten(test_crudo: pd.DataFrame) -> None:
    muestra = test_crudo.sample(n=5, random_state=0)
    esperado = predecir(muestra)
    muestra["GrLivArea"] = muestra["GrLivArea"].astype(str)

    resultado = predecir(muestra)

    pd.testing.assert_frame_equal(resultado, esperado)


def test_valores_faltantes_se_imputan(test_crudo: pd.DataFrame) -> None:
    muestra = test_crudo.sample(n=5, random_state=0)
    muestra[["LotFrontage", "BsmtQual", "GarageType"]] = np.nan

    resultado = predecir(muestra)

    assert resultado[COLUMNA_COHERENCIA].all()


def test_registro_unico_con_categoricas_nulas(
    test_crudo: pd.DataFrame,
) -> None:
    registro = test_crudo.sample(n=1, random_state=5).reset_index()
    registro[["GarageType", "GarageFinish", "BsmtQual"]] = None

    resultado = predecir_registros(registro.to_dict(orient="records"))

    assert resultado[0][COLUMNA_COHERENCIA]


def test_predecir_registros(test_crudo: pd.DataFrame) -> None:
    muestra = test_crudo.sample(n=10, random_state=3)
    registros = muestra.reset_index().to_dict(orient="records")

    resultado = predecir_registros(registros)

    assert len(resultado) == len(registros)
    assert [r[COLUMNA_ID] for r in resultado] == muestra.index.tolist()
    assert set(resultado[0]) == {
        COLUMNA_ID, COLUMNA_PREDICCION, COLUMNA_COHERENCIA
    }


def test_predecir_registros_vacio_lanza_error() -> None:
    with pytest.raises(ErrorEntradaInvalida, match="vacía"):
        predecir_registros([])


def test_evaluar_coherencia() -> None:
    predicciones = pd.Series([100000, np.nan, np.inf, 10, 2_000_000])

    resultado = evaluar_coherencia(predicciones, 34900, 755000)

    assert resultado.tolist() == [True, False, False, False, False]


def test_prediccion_no_coherente_se_marca_y_registra(
    monkeypatch: pytest.MonkeyPatch, test_crudo: pd.DataFrame
) -> None:
    real = inferencia.cargar_pipeline()
    falso = _pipeline_falso(lambda x: np.r_[1e9, real.predict(x)[1:]])
    monkeypatch.setattr(inferencia, "cargar_pipeline", lambda: falso)
    muestra = test_crudo.sample(n=5, random_state=0)

    resultado = predecir(muestra)

    assert not resultado[COLUMNA_COHERENCIA].iloc[0]
    assert resultado[COLUMNA_COHERENCIA].iloc[1:].all()
    log = RUTA_LOG.read_text(encoding="utf-8")
    assert f"no coherente para {COLUMNA_ID}={muestra.index[0]}" in log


def test_ninguna_prediccion_coherente_lanza_error(
    monkeypatch: pytest.MonkeyPatch, test_crudo: pd.DataFrame
) -> None:
    falso = _pipeline_falso(lambda x: np.full(len(x), np.nan))
    monkeypatch.setattr(inferencia, "cargar_pipeline", lambda: falso)

    with pytest.raises(ErrorPrediccion, match="Ninguna"):
        predecir(test_crudo.sample(n=5, random_state=0))


def test_fallo_del_pipeline_lanza_error(
    monkeypatch: pytest.MonkeyPatch, test_crudo: pd.DataFrame
) -> None:
    def predecir_con_error(x: pd.DataFrame) -> np.ndarray:
        raise ValueError("fallo simulado")

    falso = _pipeline_falso(predecir_con_error)
    monkeypatch.setattr(inferencia, "cargar_pipeline", lambda: falso)

    with pytest.raises(ErrorPrediccion, match="falló"):
        predecir(test_crudo.sample(n=5, random_state=0))


def test_log_registra_control_de_calidad(test_crudo: pd.DataFrame) -> None:
    predecir(test_crudo.sample(n=3, random_state=11))

    ultimas = RUTA_LOG.read_text(encoding="utf-8").splitlines()[-3:]
    assert any("Control de calidad | filas de entrada: 3" in linea
               for linea in ultimas)
