# cedear-ml

Dos preguntas distintas sobre CEDEARs, dos herramientas separadas que no comparten nada más que los datos:

| Pregunta | Módulo | Estado |
|---|---|---|
| ¿Mañana sube o baja? | `train.py` + `predict.py` | Anda de punta a punta, **sin señal** |
| ¿Cuánto vale mi cartera en 1, 3 o 5 años? | `proyeccion.py` | Anda |

La primera es un modelo de ML entrenado. La segunda no puede serlo y no lo intenta: a 5 años, en 8 años de historia entran ~1,6 ventanas independientes, así que no hay con qué entrenar. Es una simulación.

Ver `CLAUDE.md` para las reglas de dominio (por qué se predice retorno y no precio, CCL, look-ahead bias, etc.).

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Uso

Cada paso deja su resultado en disco, así que el siguiente no vuelve a bajar ni recalcular nada. Corridos en orden, van de cero a una predicción:

```bash
python -m src.fetch          # yfinance + CCL + macro + panel de BYMA   -> data/raw/
python -m src.features       # indicadores tecnicos + CCL + target      -> data/processed/
python -m src.train          # walk-forward, compara modelos, guarda    -> models/
python -m src.predict        # ranking de la cartera por prob. de suba
python -m src.proyeccion     # valor de la cartera en pesos a 1/3/5 años
pytest                       # corre los tests
```

La cartera es AAPL, MELI y KO — un CEDEAR líquido por rubro, definidos en `TICKERS` (`src/fetch.py`). Están elegidos poco correlacionados a propósito: apilar tres tecnológicas triplicaría las filas sin triplicar la información.

### Predecir de punta a punta

Si ya corriste el pipeline alguna vez, para una predicción nueva alcanza con refrescar los datos y volver a predecir:

```bash
python -m src.fetch          # sin esto predecis sobre los CSV viejos
python -m src.predict
```

`src/fetch.py` es el único módulo que toca la red: los CSV de `data/raw/` son el caché y todo lo demás corre sin internet. Por eso `predict` avisa si el último dato tiene más de 5 días.

```
Modelo: cartera_random_forest.pkl  (horizonte 5 dias)
Ultimo dato: 2026-08-14  (1 dias atras)

Probabilidad de suba a 5 dias, de mayor a menor:
  KO     52.8%   SUBE
  MELI   50.9%   SUBE
  AAPL   44.8%   BAJA
```

Con varios tickers la pregunta cambia: no es "¿sube AAPL?" sino "de estos tres, ¿cuál tiene más chances de subir?". Comparar las probabilidades entre sí es legítimo porque salen todas del mismo modelo, entrenado sobre las filas de los tres juntas.

No hace falta reentrenar para predecir: `python -m src.train` guarda el modelo en `models/*.pkl` junto con la lista de features y el horizonte, y `predict` lo levanta de ahí. Reentrenás solo cuando querés incorporar los datos nuevos al modelo.

## Resultados

**Ningún modelo le gana al baseline naive.** El baseline es "predecir siempre la clase mayoritaria", o sea gritar "sube" todos los días. Lo que importa no es el accuracy sino el `edge` = accuracy − baseline:

| Horizonte | Edge (13 feat., AAPL) | Edge (+ macro, AAPL) | Edge (+ cartera de 3) | IC |
|---|---|---|---|---|
| 1 día | +0.2% | −0.6% | **−0.3%** | +0.03 |
| 5 días | −0.8% | −1.5% | **−2.6%** | −0.02 |
| 10 días | −3.4% | −5.5% | **−3.1%** | −0.01 |
| 21 días | −4.5% | −2.5% | **−1.8%** | +0.06 |

La última columna es el pipeline actual: 19 features (indicadores técnicos, CCL, VIX, curva de tasas, tasa FED, CPI y riesgo país) sobre AAPL + MELI + KO apilados, ~6.300 filas contra las ~2.100 de un solo ticker. **Triplicar los datos no creó señal**: el edge mejora en tres horizontes y empeora en uno, pero sigue negativo en los cuatro, y el information coefficient —correlación de rangos entre la confianza del modelo y el retorno real— sigue en el rango del ruido.

El resultado a 5 días abierto por ticker muestra que tampoco es que el modelo funcione en alguno y arrastre al resto:

| Ticker | Accuracy | Baseline naive | Edge |
|---|---|---|---|
| AAPL | 54.0% | 57.5% | −3.6% |
| KO | 53.0% | 54.0% | −1.0% |
| MELI | 49.7% | 53.1% | −3.3% |

Se reporta el set completo de features y el número crudo. Elegir el subconjunto que mejor puntúa contra la validación mejoraría la tabla y no significaría nada: es sobreajustar la validación por la puerta de atrás.

Vale la pena mirar la trampa que esconde la tabla de horizontes. A 21 días el modelo acierta 58.0%, bastante más que el 52.9% de 1 día — pero el baseline también sube, a 59.8%, porque en horizontes largos las acciones suben casi siempre. **Reportar solo el accuracy habría hecho parecer mejor al modelo que en realidad está peor.** Lo mismo entre tickers: MELI y KO tienen accuracy parecida a 5 días, pero el baseline de cada uno es distinto, así que el baseline se calcula por ticker y no sobre el pool.

### Limitaciones

- Tres tickers, con datos diarios de acceso público. Que no aparezca señal es el resultado esperable: si con esto se pudiera anticipar AAPL a 5 días, ya no se podría.
- El modelo es uno solo para los tres tickers (*pooled*). La apuesta es que el patrón a aprender es común y lo que faltaba eran filas; el resultado sugiere que lo que falta no son filas.
- Las series macro mensuales (CPI, tasa FED) traen un problema propio: propagadas a diaria repiten el mismo valor ~21 días, así que cada valor identifica un mes del calendario y el árbol puede memorizar la época en vez de aprender una relación. Detalle en `docs/NOTAS.md`.
- FRED devuelve valores **revisados**, no los que se conocían en cada momento. Para datos *point-in-time* de verdad habría que usar ALFRED.
- No se tunean hiperparámetros para mejorar el número: eso sería sobreajustar la validación y cambiar un modelo honesto por uno que miente mejor.
- El pipeline está completo y validado (sin look-ahead bias, con split temporal, con la demora de publicación de cada serie macro respetada); lo que falta es señal.

## Proyección a varios años

La otra pregunta: **¿cuánto va a valer mi cartera en pesos dentro de 1, 3 o 5 años?** Editás tu tenencia en `CARTERA` (`src/proyeccion.py`) y corrés:

```bash
python -m src.fetch
python -m src.proyeccion
```

```
A 5 anios, escenario devaluacion 20%/anio
           hoy $      p10 $      p50 $      p90 $ p50 en $ de hoy      x
AAPL      24,060     39,856     90,678    208,102          36,442  3.77x
MELI      24,170     22,206     86,296    320,387          34,680  3.57x
KO        27,660     58,058     99,078    162,015          39,817  3.58x
CARTERA  980,440  2,275,228  3,775,160  6,188,522       1,517,152  3.85x
```

**La columna que importa es `p50 en $ de hoy`.** De los 3,8 millones nominales, 1,5 son ganancia real; el resto es que el peso vale menos. Un número solo escondería eso.

### Cómo funciona

Se simulan 10.000 futuros posibles remuestreando bloques de un mes de retornos históricos reales (no una distribución teórica: así se conservan las colas y los clusters de volatilidad). Los bloques se sortean **una vez para todos los tickers**, de modo que la correlación entre ellos sale de los datos sin estimarla.

El precio de partida es el que cotiza en BYMA, no `subyacente × CCL / ratio`. Con eso el ratio deja de afectar el resultado —solo se usa como control en `chequear_precios()`— y el premio o descuento del CEDEAR contra su valor teórico ya viene incluido en el precio.

### Qué se estima y qué se supone

Esta es la distinción central del módulo. Partiendo la historia al medio:

| | rentabilidad 1ª mitad | 2ª mitad | volatilidad 1ª mitad | 2ª mitad |
|---|---|---|---|---|
| AAPL | +38,6% | +15,7% | 32,7% | 28,3% |
| MELI | +32,5% | +14,9% | 53,3% | 46,8% |
| KO | +12,6% | +9,9% | 21,8% | 16,7% |
| GLOB | +46,8% | **−33,8%** | 47,8% | 52,6% |

**La volatilidad se repite; la rentabilidad no.** Globant rindió +46,8% anual durante cuatro años y −33,8% los cuatro siguientes. Medida sobre los 8 años completos da −2,2%, pero el error de esa medición (volatilidad / √años) es de ±34 puntos: el valor real está entre −36,5% y +32,1%. El número existe y no significa nada.

Por eso la volatilidad se estima de los datos y **la rentabilidad esperada y la devaluación van como supuestos explícitos** (`drift_anual`, `deval_anual`). El default de 8% anual en USD es aproximadamente el rendimiento de la bolsa americana en 100 años: una estimación con muchos más datos atrás que los 8 años de cualquiera de estos tickers.

El modelo no tiene opinión sobre ninguna empresa. Lo que sí sabe, y salió de los datos, es el riesgo: la banda de Globant es 14,8x de ancha contra 2,8x de KO.

### Limitaciones

- **La devaluación domina el resultado en pesos.** A 5 años, cambiar el supuesto de 10% a 30% anual mueve la mediana de la cartera de 2,4 a 5,6 millones. Por eso se imprime la grilla de sensibilidad: el número final depende más de ese supuesto que de las tres acciones juntas.
- El CCL entra como factor determinista, no como variable simulada. Es a propósito: no es una serie con historia proyectable (67% anual compuesto desde 2018, +3,4% en 2026), y ponerle una distribución inventada le daría al resultado una precisión falsa. La consecuencia es que la banda p10–p90 **subestima** la incertidumbre real: es solo el riesgo del subyacente en USD.
- No hay costos, impuestos ni dividendos. El spread real está en `data/raw/cedears.csv` (0,2–0,5% en los tickers de la cartera) pero todavía no se descuenta.
- El horizonte se redondea al múltiplo de 21 días más cercano (±2 semanas a 5 años).
- Los tickers están elegidos hoy sabiendo cómo les fue: cualquier resultado sobre esta canasta tiene sesgo de supervivencia.
