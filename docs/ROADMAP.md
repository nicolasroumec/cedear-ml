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

- [x] Sumar variables macro (tasa FED, riesgo país) si el modelo simple no alcanza
- [ ] Evaluar `lightgbm`/`xgboost` solo si sklearn se queda corto
- [x] Escalar de 1 CEDEAR a una cartera chica (2-3 tickers)

**Resultado de la cartera (AAPL + MELI + KO): triplicar los datos tampoco crea señal.** Con ~6.300 filas en vez de ~2.100, el edge queda en −0.3% / −2.6% / −3.1% / −1.8% a 1, 5, 10 y 21 días: mejora en tres horizontes y empeora en uno, pero sigue negativo en los cuatro, y el IC sigue entre −0.02 y +0.06. Abierto por ticker a 5 días el edge es −3.6% (AAPL), −1.0% (KO), −3.3% (MELI): tampoco es que ande en uno y los otros lo arrastren. Los tickers se eligieron de rubros poco correlacionados a propósito — apilar tres tecnológicas habría triplicado las filas sin triplicar la información. Dos decisiones de método que esto obligó: el walk-forward corta por **fecha** y no por posición de fila (si no, un mismo día quedaría repartido entre train y test), y el baseline naive se calcula **por ticker** (KO no sube el mismo porcentaje de días que MELI).

Con esto queda cerrada la hipótesis "faltan datos". Lo que falta no son filas: es información que no está en el precio diario público.

**Resultado de las variables macro: tampoco alcanzan.** Con VIX, curva de tasas, tasa FED, CPI y riesgo país (19 features en vez de 13) el edge empeora en 3 de los 4 horizontes: −0.6% / −1.5% / −5.5% / −2.5% a 1, 5, 10 y 21 días. El IC sigue entre −0.04 y +0.07, o sea ruido. Se deja el set completo de features y el número crudo, sin elegir el subconjunto que mejor puntúa: hacer eso sería sobreajustar la validación por la puerta de atrás. Ver `docs/NOTAS.md` para el detalle de por qué las series mensuales hacen ruido.

## Fase 7 — Proyección a varios años (listo)

Fase aparte y no continuación de la 5: es **otra pregunta**. Las fases 1-5
responden "¿mañana sube o baja?"; esta responde "¿cuánto vale mi cartera en
pesos dentro de 1, 3 o 5 años?", que es lo que motivó el proyecto desde el
principio. No comparte código con el modelo entrenado y no debería.

- [x] `src/fetch.py::fetch_cedears()` — panel de CEDEARs de BYMA vía data912
      (público, sin API key). Primera fuente del lado argentino y la única con
      `bid`/`ask`.
- [x] `src/proyeccion.py` — bootstrap por bloques de retornos históricos, con
      los mismos bloques para todos los tickers para conservar su correlación
- [x] Anclar el precio de hoy en el que cotiza en BYMA, no en
      `subyacente × CCL / ratio`
- [x] `chequear_precios()` — control de que ratios y precios cierran contra el CCL
- [x] Tests: que el nivel de yfinance no afecte el resultado, que los tickers
      compartan bloques, que el chequeo detecte un ratio cambiado

**Por qué no es ML.** A 5 años, en 8 años de historia entran ~1,6 ventanas
independientes. No hay con qué entrenar, y fingir que sí sería peor que no
hacerlo. Es una simulación: 10.000 futuros posibles, y la salida es un rango.

**La decisión de diseño que sostiene todo:** la volatilidad se estima de los
datos, la rentabilidad esperada y la devaluación no. Partiendo la historia al
medio, la volatilidad se repite (AAPL 32,7% → 28,3%) y la rentabilidad no
(GLOB +46,8% → −33,8%). El error de estimar rentabilidad con 8 años es
`vol/√años` = ±34 puntos para GLOB. Por eso van como parámetros explícitos y no
como estimaciones disfrazadas.

Pendiente si se retoma: descontar el spread real (ya está en
`data/raw/cedears.csv`, 0,2-0,5% en los tickers de la cartera), impuestos y
dividendos. Y decidir si `drift_anual` pasa a ser por ticker — se evaluó y se
descartó: serían opiniones sobre cada empresa escritas en el código, que a los
tres meses ya nadie recuerda que eran corazonadas. Mejor correr el escenario
suelto cuando hace falta.

## Fase 6 — Portfolio (proyecto pensado para mostrarse, ver `docs/WORKFLOW.md`)

- [ ] Docstrings + type hints en todas las funciones de `src/` (hoy los stubs no los tienen)
- [ ] EDA con visualizaciones reales en `notebooks/01_exploracion.ipynb` (no solo código, también lectura de lo que se ve)
- [ ] GitHub Actions: correr `pytest` en cada push/PR — la única pieza de "infra" que suma para portfolio sin ser peso muerto
- [ ] `LICENSE` (MIT, salvo que prefieras otra)
- [ ] Revisión final del `README.md`: que cuente la historia completa (problema → enfoque → resultados → limitaciones), no solo comandos

No se arranca una fase sin haber cerrado la anterior — evita tener features calculados sobre datos que todavía no se validaron, o un modelo entrenado sobre un target mal definido.
