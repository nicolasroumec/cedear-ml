# Roadmap — cedear-ml

Incrementos en orden. Cada uno es una rama y (idealmente) un commit por tarea chica dentro de esa rama — ver `docs/WORKFLOW.md` para la convención de ramas/commits.

## Fase 0 — Setup (listo)

- [x] Definir nombre del proyecto/repo (`cedear-ml`)
- [x] `CLAUDE.md` con reglas de dominio (target = retorno/dirección, CCL como serie propia, look-ahead bias, split temporal)
- [x] Elegir stack (Python + pandas/sklearn/pandas-ta, sin Poetry/Docker/MLflow)
- [x] Andamiaje de carpetas (`src/`, `tests/`, `notebooks/`, `data/`, `models/`) con módulos stub

## Fase 1 — Datos crudos

- [ ] `src/fetch.py::fetch_underlying(ticker)` — OHLCV de AAPL vía `yfinance`, guardar en `data/raw/`
- [ ] `src/fetch.py::fetch_ccl()` — histórico CCL vía dolarapi.com (o Bluelytics de respaldo)
- [ ] Test: los datos descargados tienen las columnas esperadas y sin huecos de fecha inexplicados

## Fase 2 — Features y target

- [ ] `src/features.py::add_technical_indicators` — SMA, EMA, RSI, MACD, Bollinger, ATR vía `pandas-ta`
- [ ] Unir el CCL al dataset del subyacente (join por fecha)
- [ ] `src/features.py::add_target` — retorno a N días + etiqueta binaria (`> 0` → 1, si no → 0)
- [ ] Test real de look-ahead bias (reemplaza el placeholder actual en `tests/test_features.py`)

## Fase 3 — Entrenamiento

- [ ] `src/train.py` — split temporal con `TimeSeriesSplit` (walk-forward), nunca random shuffle
- [ ] Entrenar `RandomForestClassifier` y `GradientBoostingClassifier`, comparar
- [ ] Métricas: accuracy, precision/recall, matriz de confusión — **y comparar contra un baseline naive** (predecir siempre la clase mayoritaria) para saber si el modelo realmente aporta algo
- [ ] Guardar el mejor modelo en `models/`

## Fase 4 — Inferencia

- [ ] `src/predict.py` — cargar modelo guardado y predecir sobre datos más recientes
- [ ] Documentar cómo correr una predicción de punta a punta (actualizar `README.md`)

## Fase 5 — Iterar (recién después de tener el pipeline completo funcionando)

- [ ] Sumar variables macro (tasa FED, riesgo país) si el modelo simple no alcanza
- [ ] Evaluar `lightgbm`/`xgboost` solo si sklearn se queda corto
- [ ] Escalar de 1 CEDEAR a una cartera chica (2-3 tickers)

No se arranca una fase sin haber cerrado la anterior — evita tener features calculados sobre datos que todavía no se validaron, o un modelo entrenado sobre un target mal definido.
