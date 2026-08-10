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

### `src/train.py` — entrenar y, sobre todo, medir honestamente

`walk_forward_splits()` genera los cortes: entrena con el pasado, testea con
el futuro inmediato, cinco veces, cada vez con más historia. Lleva
`gap=horizonte`, un hueco entre train y test: como el target de la fila `t`
usa el precio de `t+5`, sin ese hueco las últimas filas de train ya
contendrían el resultado de los primeros días de test.

`evaluar()` corre eso con un modelo y devuelve accuracy, precision, recall,
matriz de confusión e IC — **y el baseline naive de cada fold**. Lo que
reporto es el `edge` = accuracy − baseline. Solo eso significa algo.

No uso `StandardScaler`: los árboles parten por umbrales sobre cada feature
por separado, así que no les importa la escala. Si algún día pruebo
regresión logística o SVM, ahí sí hace falta.

**El resultado (Fase 3): ningún modelo le gana al baseline.**

| Horizonte | Mejor accuracy | Baseline | Edge |
|---|---|---|---|
| 1 día | 53.7% | 53.6% | +0.2% |
| 5 días | 56.9% | 57.6% | −0.8% |
| 10 días | 57.2% | 60.6% | −3.4% |
| 21 días | 59.0% | 63.5% | −4.5% |

El IC quedó entre −0.03 y +0.06, o sea ruido. Fijate la trampa que casi me
como: a 21 días el modelo acierta 59%, que suena mucho mejor que el 53.7% de
1 día. Pero el baseline también sube a 63.5%, porque en horizontes largos las
acciones suben casi siempre. **Mirar el accuracy solo me habría hecho elegir
el peor modelo.**

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

- Fase 0, 1, 2 y 3 cerradas. Rama actual: `train-model`.
- Dataset: 2138 filas, 13 features, horizonte 5 días.
- Modelo guardado: `models/aapl_random_forest.pkl` (edge −0.8%, o sea: todavía
  no le gana a nada). Sigue Fase 4: inferencia.
- Bug que encontré en el camino: `tests/test_fetch.py` pisaba
  `data/raw/aapl_ohlcv.csv` con solo 2024-en-adelante, y entrené sin darme
  cuenta con 653 filas en vez de 2138. Arreglado con un `tmp_path` en el test.
  Moraleja: si el número de filas no es el que espero, frenar y mirar.
- Nota de entorno: `pandas-ta` no existe para Python 3.11, así que los
  indicadores están escritos a mano con pandas (son one-liners de
  `rolling`/`ewm`). `jupyter` tampoco se instaló — falla por el límite de
  rutas largas de Windows; se arregla habilitando long paths cuando lo
  necesite para los notebooks.
