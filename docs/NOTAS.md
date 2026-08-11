# Notas personales — para no perderme

Mis apuntes de qué hace cada cosa y por qué. No es documentación del repo
(eso es el README), es para mí.

---

## Glosario — las palabras que se me cruzan

Escrito con el proyecto adelante: cada término apunta a dónde vive en el código.

**Pipeline.** La cadena de pasos que va del dato crudo a la predicción, donde
la salida de cada paso es la entrada del siguiente. Acá son cuatro:
`fetch → features → train → predict`. La palabra no es más que eso: no hay una
librería de "pipelines" ni nada instalado, son cuatro archivos `.py` que corro
en orden. Lo que sí importa es que cada paso **guarda en disco**: por eso puedo
correr solo `predict` sin volver a bajar dos mil filas de yfinance, y por eso
si algo sale raro puedo abrir el CSV intermedio y mirar en qué paso se rompió.
(Ojo: `sklearn` tiene una clase `Pipeline` que es otra cosa más chica —
encadenar transformaciones adentro de un modelo. Acá no la uso.)

**`.pkl` / pickle.** Pickle es el formato de Python para **serializar**:
agarrar un objeto que vive en memoria y escribirlo a disco tal cual, para
levantarlo después idéntico. Entrenar los 300 árboles del Random Forest tarda;
guardarlo en `models/aapl_random_forest.pkl` significa que `predict.py` lo
levanta en un segundo con `joblib.load()` y no reentrena nada. Adentro del pkl
guardo un diccionario con tres cosas: el modelo entrenado, la lista de features
**en orden**, y el horizonte. Las últimas dos no son decoración — ver abajo.
Dos avisos: es frágil entre versiones de sklearn (por eso el pkl no se
commitea, se regenera), y **nunca abrir un pkl ajeno**, porque des-serializar
ejecuta código arbitrario.

**Feature.** Una variable de entrada, una columna del dataset. El modelo mira
las 13 features de un día y responde. En `FEATURE_COLUMNS`.

**Target.** Lo que quiero predecir, la respuesta correcta. Acá: ¿el retorno a
5 días fue positivo? 1 o 0. En `add_target()`.

**Entrenar.** Mostrarle al modelo muchas filas con su target al lado para que
encuentre patrones. `modelo.fit(X, y)` — `X` son las features, `y` el target.

**Accuracy / precision / recall.** Accuracy = qué porcentaje de días le pegué.
Precision = de los días que dije "sube", cuántos subieron. Recall = de los días
que subieron, cuántos agarré. Se separan porque un modelo que grita "sube"
siempre tiene recall perfecto y es inútil.

**Baseline.** La respuesta tonta contra la que comparo. Acá: predecir siempre
la clase mayoritaria. Si el modelo no le gana, no aprendió nada.

**Edge.** Accuracy − baseline. **El único número que significa algo acá.**

**IC (information coefficient).** Correlación de rangos entre la confianza del
modelo y el retorno que de verdad ocurrió. Mide si ordena bien los días, no
solo si acierta el signo. Entre −0.03 y +0.06 es ruido.

**Sobreajuste (overfitting).** Cuando el modelo se aprende de memoria el ruido
del set de entrenamiento en vez del patrón, y anda bárbaro ahí y mal en datos
nuevos. La versión sutil es sobreajustar la *validación*: tocar hiperparámetros
mirando el resultado hasta que dé lindo. Por eso el edge negativo se ataca con
más variables (Fase 5) y no tuneando.

**Probabilidad / `predict_proba`.** En vez de "sube o baja", el modelo devuelve
un número entre 0 y 1. `predict()` corta en 0.5; `predict_proba()` da el número
crudo, que dice además *cuánta* confianza tiene.

**Por qué el orden de las features importa tanto.** sklearn recibe una matriz
de números y las identifica **por posición, no por nombre**. Si entrenó con
`ret_1` en la columna 0 y le paso `rsi_14` ahí, no falla ni avisa: predice
fruta con total seguridad. Por eso el orden viaja adentro del pkl y
`predict.py` lo usa siempre. Hay un test solo para eso.

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

### `src/predict.py` — usar el modelo guardado

`load_model()` levanta el pkl, `predict()` devuelve la probabilidad de suba
por fila, `predict_latest()` junta las dos cosas sobre el último día con
datos.

**Las columnas se toman de `model["features"]`, nunca del DataFrame.** sklearn
las pasa por posición, no por nombre: si el orden no coincide con el del
entrenamiento no falla ni avisa, devuelve fruta. Eso lo cubre
`test_predict_respeta_el_orden_de_features_del_bundle`.

**El detalle que me trabó**: `build_dataset` tiraba las últimas N filas
(`dropna` sobre el target) — justo las que necesito para predecir, porque su
futuro todavía no pasó. Le agregué `require_target=False` para inferencia.
Y de paso corté el `to_csv` en ese camino: si inferencia también escribiera,
pisaría `data/processed/aapl_dataset.csv` con filas sin target y el próximo
entrenamiento que lo leyera arrancaría con otros datos. Es exactamente el bug
de `test_fetch` otra vez, así que esta vez lo dejé testeado.

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

- Fase 0, 1, 2 y 3 cerradas. Rama actual: `predict-inference` (Fase 4, falta
  el README).
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
