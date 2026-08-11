# Roadmap — cedear-ml

Incrementos en orden. Cada uno es una rama y (idealmente) un commit por tarea chica dentro de esa rama — ver `docs/WORKFLOW.md` para la convención de ramas/commits.

## Fase 0 — Setup (listo)

- [x] Definir nombre del proyecto/repo (`cedear-ml`)
- [x] `CLAUDE.md` con reglas de dominio (target = retorno/dirección, CCL como serie propia, look-ahead bias, split temporal)
- [x] Elegir stack (Python + pandas/sklearn/pandas-ta, sin Poetry/Docker/MLflow)
- [x] Andamiaje de carpetas (`src/`, `tests/`, `notebooks/`, `data/`, `models/`) con módulos stub

## Fase 1 — Datos crudos

- [x] `src/fetch.py::fetch_underlying(ticker)` — OHLCV de AAPL vía `yfinance`, guardar en `data/raw/`
- [x] `src/fetch.py::fetch_ccl()` — histórico CCL vía ArgentinaDatos (`api.argentinadatos.com`, la fuente que sí tiene series históricas; dolarapi.com solo da el valor del día)
- [x] Test: los datos descargados tienen las columnas esperadas (`tests/test_fetch.py`)

## Fase 2 — Features y target

- [x] `src/features.py::add_technical_indicators` — SMA, EMA, RSI, MACD, Bollinger, ATR (a mano con pandas: `pandas-ta` no soporta Python 3.11)
- [x] Unir el CCL al dataset del subyacente (join por fecha, normalizando timezone + ffill)
- [x] `src/features.py::add_target` — retorno a N días + etiqueta binaria (`> 0` → 1, si no → 0)
- [x] Test real de look-ahead bias (reemplaza el placeholder actual en `tests/test_features.py`)

## Fase 3 — Entrenamiento

- [x] `src/train.py` — split temporal con `TimeSeriesSplit` (walk-forward, con `gap=horizonte` para que el target no se filtre al train), nunca random shuffle
- [x] Entrenar `RandomForestClassifier` y `GradientBoostingClassifier`, comparar
- [x] Métricas: accuracy, precision/recall, matriz de confusión, IC — **y comparar contra un baseline naive** (predecir siempre la clase mayoritaria) para saber si el modelo realmente aporta algo
- [x] Guardar el mejor modelo en `models/`

**Resultado: ningún modelo le gana al baseline naive.** Con horizontes de 1, 5, 10 y 21 días el edge va de +0.2% a -7.4%, y el IC está en el rango del ruido (±0.06). No es un bug: es el resultado honesto de indicadores técnicos solos sobre un único ticker. Se documenta como tal y se ataca en Fase 5 (más variables), no tuneando hiperparámetros hasta que el número quede lindo — eso sería sobreajustar la validación.

## Fase 4 — Inferencia

- [x] `src/predict.py` — cargar modelo guardado y predecir sobre datos más recientes
- [x] Documentar cómo correr una predicción de punta a punta (actualizar `README.md`)
- [x] Sección "Resultados" en `README.md` con las métricas reales del modelo vs. el baseline naive (recién acá, cuando hay resultados de verdad para mostrar)

## Fase 5 — Iterar (recién después de tener el pipeline completo funcionando)

- [ ] Sumar variables macro (tasa FED, riesgo país) si el modelo simple no alcanza
- [ ] Evaluar `lightgbm`/`xgboost` solo si sklearn se queda corto
- [ ] Escalar de 1 CEDEAR a una cartera chica (2-3 tickers)

## Fase 6 — Portfolio (proyecto pensado para mostrarse, ver `docs/WORKFLOW.md`)

- [ ] Docstrings + type hints en todas las funciones de `src/` (hoy los stubs no los tienen)
- [ ] EDA con visualizaciones reales en `notebooks/01_exploracion.ipynb` (no solo código, también lectura de lo que se ve)
- [ ] GitHub Actions: correr `pytest` en cada push/PR — la única pieza de "infra" que suma para portfolio sin ser peso muerto
- [ ] `LICENSE` (MIT, salvo que prefieras otra)
- [ ] Revisión final del `README.md`: que cuente la historia completa (problema → enfoque → resultados → limitaciones), no solo comandos

No se arranca una fase sin haber cerrado la anterior — evita tener features calculados sobre datos que todavía no se validaron, o un modelo entrenado sobre un target mal definido.
