# cedear-ml

Pipeline de ML para predecir el retorno/dirección de CEDEARs (no el precio absoluto). Estado actual: solo documentación de referencia, sin código todavía.

Guía completa para humanos (mecánica del instrumento, indicadores, fuentes, trampas de ML): https://claude.ai/code/artifact/dbcddce8-fd01-4ef2-ac05-e27aea564962

## Reglas de dominio que no son negociables

- **Precio del CEDEAR = precio_subyacente_USD × ratio × CCL.** Son dos series independientes (mercado de EEUU + tipo de cambio implícito argentino). No tratar el precio en pesos como una sola señal.
- **CCL** = precio en pesos de una acción dual-listada (ej. GGAL) / precio en USD ajustado por ratio. Se necesita como serie propia, no derivarlo solo del CEDEAR que se está prediciendo.
- **Target: retorno o dirección, nunca precio absoluto.** Un modelo entrenado contra el nivel de precio aprende a copiar el valor de hoy (el precio es un random walk aproximado) y da métricas engañosamente buenas sin ser útil.
- **Usar adjusted close** del subyacente, no el crudo — splits/dividendos generan saltos falsos si no.
- **Sin look-ahead bias:** ningún feature en la fila de fecha `t` puede depender de datos posteriores a `t`. Incluye medias/std usadas para normalizar.
- **Split temporal (walk-forward), nunca random shuffle** entre train/val — mezclar al azar filtra el futuro al pasado.
- **Normalización:** estadísticas (media, desvío) calculadas solo sobre el set de entrenamiento, aplicadas después al de validación.
- **Ratio de conversión del CEDEAR cambia** por ajustes del banco depositario — verificar el vigente antes de construir series largas; un cambio de ratio no ajustado se ve como un salto de precio que no es señal de mercado.

## Fuentes de datos previstas

| Dato | Fuente |
|---|---|
| OHLCV subyacente, índices (S&P 500) | `yfinance` |
| Tasa FED, CPI EEUU | FRED |
| Tasa/oficial Argentina | BCRA (API pública) |
| CCL histórico | ArgentinaDatos (`api.argentinadatos.com/v1/cotizaciones/dolares/contadoconliqui`) — dolarapi.com solo da el valor del día, no serie histórica |
| Cotización CEDEAR en pesos, ratio vigente | IOL / Rava / ByMA Data |

## Stack y convenciones de código

- **Entorno**: Python 3.11+, `venv` + `requirements.txt`. Sin Poetry/uv — overhead innecesario para un proyecto de una persona.
- **Librerías**: `yfinance` (subyacente/índices), `requests` (CCL/BCRA/FRED), `pandas`/`numpy`, `pandas-ta` (indicadores técnicos — no reinventarlos a mano), `scikit-learn` (modelos + `TimeSeriesSplit` para walk-forward), `matplotlib`, `jupyter`. `lightgbm`/`xgboost` solo si sklearn se queda corto — no instalar por adelantado.
- **Modelo inicial**: clasificación binaria (dirección del retorno a N días) con `RandomForestClassifier`/`GradientBoostingClassifier`. Sin LSTM/deep learning todavía — no hay evidencia de que la complejidad pague con el volumen de datos disponible por CEDEAR.
- **Alcance inicial**: un solo CEDEAR (AAPL) para validar el pipeline end-to-end antes de escalar a una cartera.
- **Estructura**: `src/fetch.py` (descarga), `src/features.py` (indicadores + target), `src/train.py` (walk-forward + entrenamiento), `src/predict.py` (inferencia) — módulos simples ejecutables, sin clases hasta que haya una razón concreta. `notebooks/` es solo para exploración, la lógica reusable vive en `src/`. `tests/` con smoke tests, en particular contra look-ahead bias.
- Explícitamente afuera hasta que el proyecto lo justifique: Docker, MLflow, orquestador (Airflow/Prefect), API REST, empaquetado pip.
