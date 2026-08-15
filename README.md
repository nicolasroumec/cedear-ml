# cedear-ml

Pipeline de ML para predecir el retorno/dirección de CEDEARs. Ver `CLAUDE.md` para las reglas de dominio (por qué se predice retorno y no precio, CCL, look-ahead bias, etc.).

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Uso

Cada paso deja su resultado en disco, así que el siguiente no vuelve a bajar ni recalcular nada. Corridos en orden, van de cero a una predicción:

```bash
python -m src.fetch          # OHLCV de yfinance + CCL de ArgentinaDatos -> data/raw/
python -m src.features       # indicadores tecnicos + CCL + target      -> data/processed/
python -m src.train          # walk-forward, compara modelos, guarda    -> models/
python -m src.predict        # ranking de la cartera por prob. de suba
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
