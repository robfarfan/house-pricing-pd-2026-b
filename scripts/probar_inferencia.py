"""Prueba la inferencia con observaciones aleatorias de ``test.csv``.

Toma ``n`` casas al azar del set de test crudo, ejecuta ``predecir`` y
muestra las predicciones con el resumen del control de calidad. Los
mensajes también quedan en ``scripts/inferencia.log``.

Uso::

    python scripts/probar_inferencia.py --n 20 --semilla 42
"""

import argparse

import pandas as pd

from house_pricing.config import RAW_DATA_DIR
from inferencia import (
    COLUMNA_COHERENCIA,
    COLUMNA_ID,
    RUTA_LOG,
    predecir,
)


def tomar_muestra(n: int, semilla: int | None = None) -> pd.DataFrame:
    """Toma casas aleatorias del set de test crudo.

    Args:
        n: Número de casas a tomar.
        semilla: Semilla aleatoria. Con ``None`` cada ejecución toma
            una muestra distinta.

    Returns:
        DataFrame con ``n`` filas de ``test.csv`` e ``Id`` como índice.

    Raises:
        ValueError: Si ``n`` es menor que 1 o mayor que el número de
            filas de ``test.csv``.
    """
    test = pd.read_csv(RAW_DATA_DIR / "test.csv", index_col=COLUMNA_ID)
    if not 1 <= n <= len(test):
        raise ValueError(f"n debe estar entre 1 y {len(test)}")
    return test.sample(n=n, random_state=semilla)


def main() -> None:
    """Ejecuta la prueba desde la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Prueba la inferencia con casas aleatorias de test."
    )
    parser.add_argument(
        "--n", type=int, default=20, help="Número de casas (20 por defecto)."
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=None,
        help="Semilla aleatoria para repetir la misma muestra.",
    )
    argumentos = parser.parse_args()

    muestra = tomar_muestra(argumentos.n, argumentos.semilla)
    resultado = predecir(muestra)

    coherentes = int(resultado[COLUMNA_COHERENCIA].sum())
    print(resultado.round(0).to_string())
    print(
        f"\nFilas de entrada: {len(muestra)} | "
        f"filas de salida: {len(resultado)} | "
        f"coherentes: {coherentes} | "
        f"no coherentes: {len(resultado) - coherentes}"
    )
    print(f"Log: {RUTA_LOG}")


if __name__ == "__main__":
    main()
