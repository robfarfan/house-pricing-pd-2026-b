# House_Pricing: estado del proyecto y próximos pasos

Última actualización: 2026-10-30

Este documento resume lo que se hizo hasta ahora para retomar el trabajo en la siguiente sesión.

## Objetivo

Construir un modelo que prediga `SalePrice` con el dataset de Kaggle *House Prices* y servirlo con una API de FastAPI.

## Estado actual

- ✅ Estructura del proyecto con Cookiecutter Data Science v2.
- ✅ Entorno conda `env_house_pricing` (Python 3.11) funcionando.
- ✅ Análisis exploratorio en `notebooks/1.0-rf-analisis-exploratorio.ipynb`.
- ✅ Features seleccionadas en `pipeline_config.yaml` (`MODEL_FEATURES`).
- ✅ Ingeniería de características en `notebooks/2.0-rf-feature-engineering.ipynb`, con un pipeline de scikit-learn y feature-engine.
- ✅ Torneo de modelos en `notebooks/3.0-rf-entrenamiento-modelos.ipynb`. El ganador es **GradientBoosting**.
- ✅ Pipeline de inferencia completo, de los datos crudos al precio en dólares, en `models/pipeline_inferencia.joblib`.
- ✅ Notebook de inferencia en `notebooks/4.0-rf-inferencia.ipynb`, con las predicciones en `data/processed/predicciones_test.csv`.
- ✅ Script de inferencia para la API en `scripts/inferencia.py`, con logging, control de calidad y pruebas.
- ✅ Repositorio git público en https://github.com/robfarfan/house-pricing-pd-2026-b (`main`).
- 🔄 Integración de MLflow en los notebooks 3.0 y 4.0, en la rama `feature/mlflow` (sin commit todavía; ver sección MLflow).
- ⏭️ **Siguiente paso:** crear la API con FastAPI que consuma `scripts/inferencia.py` (`predecir_registros`).

## Estructura relevante

```
House_Pricing/
├── pipeline_config.yaml       <- MODEL_FEATURES (30 features) y SALEPRICE_RANGE (rango de coherencia)
├── CONTINUAR_SESION.md        <- este documento
├── requirements.txt           <- dependencias con versión fija
├── environment.yml            <- generado por la plantilla (desactualizado, ver Pendientes)
├── pyproject.toml             <- requires-python ">=3.11" y configuración de pytest (pythonpath = scripts)
├── data/raw/                  <- train.csv, test.csv, sample_submission.csv, data_description.txt
├── data/processed/            <- X_train, X_val, X_test, y_train, y_val (salida del 2.0) y predicciones_test.csv
├── models/
│   ├── pipeline_fe.joblib         <- pipeline de features ajustado con el 80 % (notebook 2.0)
│   ├── modelo_ganador.joblib      <- features + GradientBoosting ajustado con el 80 % (el evaluado en el torneo)
│   └── pipeline_inferencia.joblib <- PIPELINE DE PRODUCCIÓN: crudo → USD, ajustado con el 100 % de train
├── notebooks/
│   ├── 1.0-rf-analisis-exploratorio.ipynb
│   ├── 2.0-rf-feature-engineering.ipynb
│   ├── 3.0-rf-entrenamiento-modelos.ipynb
│   └── 4.0-rf-inferencia.ipynb
├── scripts/
│   ├── inferencia.py          <- módulo de inferencia para FastAPI
│   ├── probar_inferencia.py   <- prueba por terminal con casas aleatorias de test
│   └── inferencia.log         <- log único de todas las ejecuciones
├── tests/test_inferencia.py   <- 21 pruebas de pytest
└── house_pricing/             <- paquete del proyecto (instalado en modo editable)
    ├── config.py              <- rutas: RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PROJ_ROOT, etc.
    ├── dataset.py, features.py, plots.py    <- todavía con el código de ejemplo de la plantilla
    └── modeling/{train.py, predict.py}      <- todavía con el código de ejemplo de la plantilla
```

## Datos

| Archivo | Filas | Columnas | Nota |
|---|---|---|---|
| `train.csv` | 1460 | 81 | Incluye el target `SalePrice` (de 34,900 a 755,000 USD) |
| `test.csv` | 1459 | 80 | Sin `SalePrice` |

`data_description.txt` explica cada variable. En muchas columnas categóricas, un valor faltante significa "no tiene" (por ejemplo, `PoolQC`, `Alley` o `FireplaceQu`).

## Entorno

- **Nombre:** `env_house_pricing`, con Python 3.11.18.
- **Conda no está en el PATH.**
  - Ejecutable de conda: `C:\Users\rfarfan\AppData\Local\anaconda3\Scripts\conda.exe`.
  - Python del entorno: `C:\Users\rfarfan\AppData\Local\anaconda3\envs\env_house_pricing\python.exe`.
- **Instalar una librería:** `conda install -n env_house_pricing -c conda-forge <paquete>` y después agregarla a `requirements.txt` con `==<versión>`, en orden alfabético.
- **Librerías principales:** feature-engine 1.9.4, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.4.6, seaborn 0.13.2, pyyaml 6.0.3, pytest 9.2.0 y loguru 0.7.3.
- **El paquete del proyecto** está instalado en modo editable (`pip install -e .`), así que `from house_pricing.config import ...` funciona desde los notebooks y los scripts.
- **⚠️ BLAS:** el entorno debe quedarse en OpenBLAS (`libblas=*=*openblas`). Con MKL, el kernel de VS Code se cae con el error `0xc06d007f` en cualquier operación matricial, porque MKL necesita el entorno activado. Después de cada `conda install`, revisa que `libblas` siga en openblas.
- **Para ejecutar los notebooks en VS Code**, selecciona el kernel `env_house_pricing`.
- **`conda run`** no acepta scripts de varias líneas con `-c`. Escribe el script en un archivo y ejecútalo.

## Convenciones y forma de trabajo

- **Nombres de notebooks** (convenio de Cookiecutter): `<número>-rf-<descripción-con-guiones>.ipynb`.
  - Una fase por número entero: `1.0` análisis exploratorio, `2.0` feature engineering, `3.0` entrenamiento de modelos y `4.0` inferencia. Las variantes de una misma fase usan `x.1`, `x.2`…
  - El 2026-11-10 se renombraron (antes: `1.1-rf-transformaciones`, `2.0-rf-modelado` y `3.0-rf-inferencia`). Los runs de MLflow anteriores conservan el nombre viejo como origen.
- **En los notebooks:**
  - Cada paso lleva su título y una descripción simple, y va en su propia celda.
  - Ante cualquier decisión metodológica, el usuario prefiere que le preguntemos con opciones y una recomendación, en vez de asumir.
- **`requirements.txt`** se actualiza cada vez que se agrega una librería al entorno.
- **Código Python:**
  - PEP 8 a 79 columnas (se revisa con `pycodestyle`).
  - Funciones con tipos en parámetros y retornos, y docstrings con `Args`, `Returns` y `Raises`.
- **Notebooks abiertos en VS Code:** si Claude modifica un notebook que está abierto, no lo guardes. Ejecuta `Ctrl+Shift+P` → **File: Revert File** y después **Restart** + **Run All**. Si lo guardas, VS Code sobrescribe los cambios con su copia en memoria. Por eso Claude escribe los notebooks de una sola vez, con un script que genera el JSON, y los verifica ejecutándolos en un kernel.

## Notebook 1.0: análisis exploratorio

1. Carga de `train` y `test` desde `RAW_DATA_DIR`. Se usa `df = train`.
2. Estructura, estadísticas descriptivas, valores faltantes, histogramas y matriz de correlación.
3. **Discretas vs continuas:** una variable es discreta si tiene 15 valores únicos o menos. `PoolArea` es una excepción y se trata como continua. Resultado: 14 discretas y 22 continuas.
4. **Continuas:** scatter plot contra `SalePrice` con la correlación de Pearson en el título.
5. **Discretas:** boxplot con stripplot y la correlación de Pearson.
6. **Categóricas (43):** boxplot con stripplot y la **razón de correlación η**, donde η = √(SS_entre / SS_total), va de 0 a 1 y η² es el R² de un ANOVA de un factor.
7. **Selección de features:** top 10 de cada grupo. Para continuas y discretas se ordena por |r| y para categóricas por η.

### Features seleccionadas (`MODEL_FEATURES`)

| Continuas (\|r\|) | Discretas (\|r\|) | Categóricas (η) |
|---|---|---|
| GrLivArea (0.71) | OverallQual (0.79) | Neighborhood (0.74) |
| GarageArea (0.62) | GarageCars (0.64) | ExterQual (0.69) |
| TotalBsmtSF (0.61) | FullBath (0.56) | BsmtQual (0.68) |
| 1stFlrSF (0.61) | TotRmsAbvGrd (0.53) | KitchenQual (0.68) |
| YearBuilt (0.52) | Fireplaces (0.47) | GarageFinish (0.55) |
| YearRemodAdd (0.51) | HalfBath (0.28) | FireplaceQu (0.54) → descartada en el 2.0 |
| GarageYrBlt (0.49) | BsmtFullBath (0.23) | Foundation (0.51) |
| MasVnrArea (0.48) | BedroomAbvGr (0.17) | GarageType (0.50) |
| BsmtFinSF1 (0.39) | KitchenAbvGr (−0.14) | BsmtFinType1 (0.46) |
| LotFrontage (0.35) | MSSubClass (−0.08) | HeatingQC (0.44) |

## Notebook 2.0: ingeniería de características

Decisiones que tomó el usuario:

1. **Antes del pipeline:**
   - Se eliminan 2 casas con `GrLivArea > 4000` y `SalePrice < 300000`.
   - El target pasa a `SalePrice_log = log1p(SalePrice)`.
   - Se divide en 80/20 con `random_state=42`.
2. **Imputación**, según el % de faltantes en train:
   - Numéricas con ≤ 5 %: mediana.
   - Numéricas con más de 5 % y hasta 20 %: `RandomSampleImputer` (`LotFrontage`, `GarageYrBlt`).
   - Variables con más de 20 %: se descartan (`FireplaceQu`).
   - Categóricas: moda.
3. **Codificación:**
   - Ordinal con su orden real en `ExterQual`, `BsmtQual`, `KitchenQual` y `HeatingQC` (Po<Fa<TA<Gd<Ex), `GarageFinish` (Unf<RFn<Fin) y `BsmtFinType1` (Unf<LwQ<Rec<BLQ<ALQ<GLQ).
   - Nominales (`Neighborhood`, `Foundation`, `GarageType`): `RareLabelEncoder` (< 5 %) y después one-hot.
   - `MSSubClass` se mantiene como numérica discreta.
4. **Outliers:** `Winsorizer` con IQR (1.5) en las 10 continuas.
5. **Transformación:** Yeo-Johnson en las continuas con `|skew| > 0.75` después del capping. Solo quedó `MasVnrArea`.
6. **Escalado:** `StandardScaler` en todas las columnas.

**Resultado:** 44 columnas finales. El pipeline quedó guardado en `models/pipeline_fe.joblib`.

## Notebook 3.0: torneo de modelos

**Decisiones que tomó el usuario:**
- **Fuga de datos:** se parte de los datos crudos con el mismo split del 2.0 (el notebook verifica que el set de validación sea idéntico). Cada modelo es `Pipeline(clone(pipeline_fe) + modelo)`, así que las features se reajustan en cada fold.
- **Búsqueda:** `GridSearchCV` con KFold k=10 (`shuffle=True`, `random_state=42`).
- **Métrica de la CV:** el RMSE real en dólares, con `expm1` aplicado dentro del scorer y las predicciones log limitadas a [0, 15].
- **Comparación:** contra el set de validación, con el RMSE real y el NRMSE = RMSE / (máx − mín de SalePrice en validación).
- **Gráficas:** comparación del torneo (barras de RMSE y NRMSE más el boxplot de los 10 folds) y predicho vs real por modelo.

| Modelo | RMSE train | RMSE CV | RMSE val | NRMSE val | Brecha train-CV |
|---|---|---|---|---|---|
| **GradientBoosting** | 17,265 | 25,338 ± 3,316 | **22,992** | **0.0552** | 32 % |
| Lasso | 25,185 | 25,759 ± 4,622 | 23,565 | 0.0566 | 2 % |
| ElasticNet | 25,176 | 25,753 ± 4,610 | 23,585 | 0.0566 | 2 % |
| Ridge | 25,180 | 25,705 ± 4,622 | 23,652 | 0.0568 | 2 % |
| MLP | 15,350 | 26,533 ± 3,046 | 24,016 | 0.0576 | 42 % |
| RandomForest | 10,546 | 26,716 ± 4,787 | 24,232 | 0.0582 | 61 % |

**Ganador, confirmado por el usuario:** GradientBoosting con `learning_rate=0.03`, `max_depth=3`, `n_estimators=300` y `subsample=0.8`.

**Pipeline de inferencia** (sección final del 3.0), reentrenado con el 100 % de train (1458 casas) y guardado en `models/pipeline_inferencia.joblib`:

```
TransformedTargetRegressor(func=log1p, inverse_func=expm1)
└── Pipeline
    ├── seleccion  → FunctionTransformer(itemgetter(MODEL_FEATURES))
    ├── features   → clone(pipeline_fe)
    └── modelo     → GradientBoosting ganador
```

## Notebook 4.0: inferencia

1. Carga el champion del Model Registry (`models:/house_pricing@champion`, flavor `sklearn`) y lee `test.csv` crudo.
2. Predice en dólares: 1459 casas, sin faltantes ni valores negativos. La media predicha es 178,185 USD, contra 180,921 USD en train.
3. Compara en un histograma la distribución predicha con la real.
4. Guarda `data/processed/predicciones_test.csv` con `Id` y `SalePrice`, en el formato de `sample_submission.csv`.
5. Registra un run en el experimento `house_pricing-inferencia` con la versión usada, estadísticas de las predicciones, cuántas quedan fuera de `SALEPRICE_RANGE`, el CSV y el histograma.

## MLflow (rama `feature/mlflow`)

- **Backend:** SQLite local `mlflow.db` (tracking + Model Registry) y artefactos en `mlruns/<experimento>`, ambos en la raíz y fuera de git. Configuración en `house_pricing/config.py` (`MLFLOW_*`, redirigibles con las variables de entorno `MLFLOW_TRACKING_URI` y `MLFLOW_ARTIFACTS_URI`) y `house_pricing/tracking.py` (`configurar_mlflow`).
- **UI:** `mlflow ui --backend-store-uri sqlite:///mlflow.db` desde la raíz → http://127.0.0.1:5000.
- **Notebook 3.0:** jerarquía de 3 niveles. El run `torneo_AAAAMMDD_HHMMSS` agrupa la ejecución: queda abierto desde "Configuración de MLflow" hasta "Cierre del run del torneo", donde recibe `tabla_torneo.csv`, las gráficas del torneo, los tags de champion/challengers con sus versiones y el RMSE/NRMSE del champion. Dentro hay un run por modelo (parámetros, métricas train/CV/validación, `diagnostico.png`, `configuraciones.csv`, tag `posicion`) y, dentro de cada modelo, un run por configuración (42 en total). Todos llevan el tag `torneo`.
- **Notebook 4.0:** el run de inferencia se llama `inferencia_AAAAMMDD_HHMMSS`.
- **Registry:** modelo `house_pricing`. Los 3 mejores por RMSE de validación se registran como pipeline de inferencia completo reentrenado con 1,458 casas, dentro del run de su modelo: `champion` (GradientBoosting), `challenger_1` (Lasso), `challenger_2` (ElasticNet). Cada corrida del 2.0 crea versiones nuevas y mueve los aliases.
- **Detalles técnicos:** MLflow 3.18 usa `skops` por defecto, que no puede serializar feature-engine, así que se registra con `serialization_format="cloudpickle"`. La firma declara las 30 features como opcionales y los numéricos como `double`; por eso, con el flavor `pyfunc`, los numéricos deben enviarse como float (relevante para FastAPI). Ojo con el límite de 260 caracteres de rutas en Windows: las rutas de artefactos llegan a ~205.

## Script de inferencia (`scripts/inferencia.py`)

**Funciones:** todas tienen tipos y docstrings.

| Función | Para qué sirve |
|---|---|
| `predecir(datos: pd.DataFrame) -> pd.DataFrame` | Recibe datos crudos con el formato de `test.csv` y devuelve `SalePrice` y `es_coherente`. El índice se usa como `Id` |
| `predecir_registros(registros: list[dict]) -> list[dict]` | **Es la que usará FastAPI:** recibe y devuelve JSON. Usa la clave `Id` si viene en los registros |
| `validar_entrada`, `evaluar_coherencia` | Validación de la entrada y criterio de coherencia |
| `cargar_pipeline`, `cargar_configuracion` | Cargan el pipeline y la configuración, en caché con `lru_cache` |

**Validación y errores:**
- **`ErrorEntradaInvalida`:** se lanza si la entrada no es un DataFrame, está vacía, le faltan columnas o trae texto en columnas numéricas. Si los números llegan como texto, se convierten.
- **Faltantes:** se permiten, porque el pipeline los imputa, y se registran como advertencia.
- **Coherencia:** una predicción es coherente si es finita y está dentro de `SALEPRICE_RANGE` (34,900 a 755,000 USD). Las no coherentes se registran como ERROR con su `Id` y se marcan en la salida.
- **`ErrorPrediccion`:** se lanza si el pipeline falla, si el número de predicciones no coincide con el de filas o si ninguna predicción es coherente.

**Log:** usa loguru y escribe en un único archivo, `scripts/inferencia.log`, con fecha y hora en cada línea. Por cada ejecución deja la línea `Control de calidad | filas de entrada | predicciones | coherentes | no coherentes | tiempo`.

**Detalle técnico:** si una columna categórica llega sin ningún valor (por ejemplo, una sola casa sin garaje), el script la convierte al tipo `"str"` de pandas. Con `object` no basta, porque los imputadores de feature-engine llaman a `infer_objects()` y la regresan a float.

**Cómo usarlo y probarlo** (desde `House_Pricing`, con el Python del entorno):

```powershell
python scripts/inferencia.py --entrada data/raw/test.csv --salida <csv>   # inferencia sobre un CSV
python scripts/probar_inferencia.py --n 20 --semilla 42                   # casas aleatorias de test
pytest                                                                    # 21 pruebas
```

## Pendientes y decisiones abiertas

- **API con FastAPI (siguiente paso):** consumirá `predecir_registros`. Antes de empezar, hay que preguntarle al usuario por:
  - los endpoints,
  - el esquema de entrada (por ejemplo, Pydantic con las 29 features o el registro crudo completo),
  - la forma de responder a los errores (códigos HTTP para `ErrorEntradaInvalida` y `ErrorPrediccion`),
  - la ubicación de la app.
  
  Hay que instalar `fastapi` y `uvicorn` y agregarlos a `requirements.txt`.
- **Logs mezclados:** el log de pruebas está mezclado con el de producción en `scripts/inferencia.log`, que ya tiene ~750 líneas en buena parte de pytest. Hay que evaluar si se separan.
- **Advertencia de feature-engine:** queda una `Pandas4Warning` interna del `OneHotEncoder` (`inplace` en `DataFrame.drop`). No afecta los resultados; se resolverá al actualizar feature-engine.
- **Posibles mejoras del modelo:**
  - GradientBoosting sobreajusta más que Lasso (32 % contra 2 %) y gana por apenas un 2.4 %.
  - Varias mejores configuraciones quedaron en el borde de su grilla: Lasso y ElasticNet (`alpha=0.0001`), RandomForest (`max_features=0.33`) y GradientBoosting (`learning_rate=0.03`, `n_estimators=300`). Ampliar esas grillas podría mejorar los resultados.
  - El RMSE de validación del ganador es algo optimista, porque el ganador se eligió con ese mismo set.
- **Features que no se implementaron:**
  - Ideas del plan original: banderas binarias para variables con muchos ceros, antigüedades, `TotalSF` y Spearman.
  - Variables redundantes: `GarageCars`/`GarageArea`, `TotalBsmtSF`/`1stFlrSF`, `GrLivArea`/`TotRmsAbvGrd` y `YearBuilt`/`GarageYrBlt`.
  - Discretas débiles: `BedroomAbvGr`, `KitchenAbvGr` y `MSSubClass`.
- **Capping de años:** el Winsorizer también recorta `YearBuilt` y `GarageYrBlt`, aunque a menos del 1 % de las casas. El usuario todavía no decidió si excluirlos.
- **`environment.yml`** sigue como lo generó la plantilla (`name: House_Pricing`, Python 3.12) y no corresponde con el entorno real.
- **`house_pricing/features.py` y `modeling/`** siguen con el código de ejemplo de la plantilla. La lógica real está en los notebooks y en `scripts/`.
