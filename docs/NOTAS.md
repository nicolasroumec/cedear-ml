# Notas personales — para no perderme

Mis apuntes de qué hace cada cosa y por qué. No es documentación del repo
(eso es el README), es para mí.

---

## El flujo, de punta a punta

```
yfinance (AAPL, USD)  ─┐
                       ├─> src/fetch.py ──> data/raw/*.csv
ArgentinaDatos (CCL)  ─┘                        │
                                                v
                                        src/features.py ──> data/processed/aapl_dataset.csv
                                                                    │
                                                                    v
                                                            src/train.py ──> models/*.pkl
                                                                    │
                                                                    v
                                                            src/predict.py ──> predicción
```

Cada paso guarda en disco. Así el siguiente no vuelve a bajar ni recalcular
nada, y puedo correr un solo módulo sin correr todo.

---

## Archivos

### `src/fetch.py` — bajar los datos crudos

Dos funciones:

- `fetch_underlying("AAPL")` → OHLCV diario de yfinance, con `auto_adjust=True`
  (o sea Close ya ajustado por splits y dividendos, que es lo que quiero: sin
  eso, un split 4:1 se ve como una caída del 75% que no fue una caída).
- `fetch_ccl()` → serie histórica del dólar CCL. Uso ArgentinaDatos porque
  dolarapi.com solo da el valor de hoy, no la serie.

Las dos guardan un CSV en `data/raw/`. **Ese CSV es el caché**: todo lo demás
del pipeline corre sin internet.

Es el único archivo que toca la red. A propósito: si mañana una API cambia,
sé exactamente dónde mirar.

### `src/features.py` — convertir precios en variables aprendibles

Acá pasa lo interesante. Cuatro funciones:

- `add_technical_indicators(df)` → los indicadores.
- `add_ccl(df, ccl)` → pega la serie del CCL al dataset, por fecha.
- `add_target(df, horizon_days)` → lo que quiero predecir.
- `build_dataset()` → orquesta las tres y guarda el resultado.

**Todos los features son relativos, ninguno es un nivel de precio.** Esto me
costó entenderlo: si le doy `sma_20 = 187.34` al modelo, ese número significa
cosas distintas en 2018 (AAPL a 40) y en 2026 (AAPL a 310). No hay nada
estable para aprender. En cambio `close / sma_20 - 1 = 0.02` significa
"el precio está 2% arriba de su media" y eso es comparable siempre.

Los que calculo hoy:

| Feature | Qué mide |
|---|---|
| `ret_1`, `ret_5` | Cuánto se movió el precio en 1 y 5 días |
| `sma_20_ratio` | Qué tan lejos está el precio de su media simple de 20 días |
| `ema_12_ratio` | Igual pero con media exponencial (le da más peso a lo reciente) |
| `rsi_14` | Fuerza de las subas contra las bajas, 0-100. Arriba de 70 "sobrecomprado" |
| `macd_norm`, `macd_hist_norm` | Diferencia entre dos medias exponenciales = cambio de tendencia |
| `bb_pct_b` | Dónde está el precio dentro de las bandas de Bollinger (0 = piso, 1 = techo) |
| `bb_width` | Ancho de las bandas = cuánta volatilidad hubo últimamente |
| `atr_14_pct` | Volatilidad real en %, contando los gaps entre cierre y apertura |
| `volume_ratio` | Volumen de hoy contra el promedio. 1.0 = día normal, 3.0 = pasó algo |
| `ccl_ret_1`, `ccl_ret_5` | Cuánto se movió el dólar CCL |

**Por qué el CCL va como feature aparte**: el precio del CEDEAR en pesos es
`precio_USD × ratio × CCL`. Son dos series independientes (el mercado de
EEUU y el tipo de cambio argentino). Si le doy solo el precio en pesos, el
modelo no puede distinguir "AAPL subió" de "se disparó el dólar".

**El join del CCL tiene una trampa**: yfinance devuelve el índice con
timezone de Nueva York (`2026-08-07 00:00:00-04:00`) y ArgentinaDatos con
fechas planas (`2026-08-07`). Sin normalizar, el join no matchea *ninguna*
fila y quedaría todo NaN. Eso lo hace `_to_date_index()`.

Además el CCL tiene filas de fin de semana (repite el último valor) y a
veces falta un feriado argentino que sí es día de rueda en EEUU → `ffill`,
que copia el valor del día anterior. Nunca hacia atrás: copiar el valor del
día *siguiente* sería meter futuro en el pasado.

**El target** (`add_target`): retorno entre `t` y `t+N`, y la etiqueta
binaria (1 si es positivo, 0 si no). Es la única función del módulo que mira
al futuro, y tiene que ser así — es exactamente lo que el modelo debe
predecir. Las últimas N filas quedan en NaN porque ese futuro todavía no
pasó.

Al final `build_dataset` tira las filas incompletas: las primeras (los
rolling de 20 días todavía no se llenaron) y las últimas (sin futuro).

### `src/train.py` — Fase 3, todavía stub

Va a hacer walk-forward con `TimeSeriesSplit` y comparar RandomForest contra
GradientBoosting. Lo importante: comparar contra el baseline naive.

### `src/predict.py` — Fase 4, todavía stub

### `tests/test_fetch.py`

Pega a las APIs de verdad. No testea pandas, testea que las fuentes no me
cambiaron el formato — que es la forma más común de que esto se rompa sin
que me entere.

### `tests/test_features.py`

El que importa es `test_features_no_look_ahead`. La idea:

> Calculo los features dos veces: una sobre la serie completa, otra sobre la
> serie **cortada** en el día `t`. Los valores de la fila `t` tienen que dar
> **idénticos**. Si algún indicador estuviera usando datos posteriores a `t`,
> darían distinto y el test falla.

Ya lo probé rompiéndolo a propósito (normalizando con la media de toda la
serie) y efectivamente falla. Un test que no puede fallar no sirve.

Los otros tres: que el target sí mire adelante, que el ffill del CCL no mire
atrás, y que RSI/ATR/volumen den valores en rangos posibles.

---

## Cosas que me quiero acordar

- **Look-ahead bias**: cualquier dato de la fila `t` que use info posterior a
  `t`. Incluye normalizar con la media/desvío de toda la serie. Da métricas
  espectaculares y un modelo inútil en la realidad.
- **Nunca `train_test_split` con shuffle** en series temporales: mezclar al
  azar pone días de 2026 en train y días de 2019 en validación. El modelo
  "aprende" del futuro.
- **Baseline naive**: 53.6% de los días son positivos. Un modelo que acierta
  54% no está aprendiendo nada, está diciendo "sube" siempre. Ese es el
  número a superar en Fase 3.
- **El ratio del CEDEAR cambia** (lo ajusta el banco depositario). Todavía no
  lo estoy usando, pero cuando arme el precio en pesos hay que verificar el
  vigente o va a aparecer un salto que no es del mercado.

## Estado

- Fase 0, 1 y 2 cerradas. Rama actual: `features-target`.
- Dataset: 2142 filas, 13 features, 2018-01-30 → 2026-08-07.
- Sigue Fase 3: entrenamiento.
- Nota de entorno: `pandas-ta` no existe para Python 3.11, así que los
  indicadores están escritos a mano con pandas (son one-liners de
  `rolling`/`ewm`). `jupyter` tampoco se instaló — falla por el límite de
  rutas largas de Windows; se arregla habilitando long paths cuando lo
  necesite para los notebooks.
