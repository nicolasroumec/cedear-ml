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
python -m src.predict        # probabilidad de suba del ultimo dia
pytest                       # corre los tests
```

### Predecir de punta a punta

Si ya corriste el pipeline alguna vez, para una predicción nueva alcanza con refrescar los datos y volver a predecir:

```bash
python -m src.fetch          # sin esto predecis sobre los CSV viejos
python -m src.predict
```

`src/fetch.py` es el único módulo que toca la red: los CSV de `data/raw/` son el caché y todo lo demás corre sin internet. Por eso `predict` avisa si el último dato tiene más de 5 días.

```
Modelo: aapl_random_forest.pkl  (horizonte 5 dias)
Ultimo dato: 2026-08-10  (1 dias atras)

Probabilidad de suba a 5 dias: 42.8%
Direccion predicha: BAJA
```

No hace falta reentrenar para predecir: `python -m src.train` guarda el modelo en `models/*.pkl` junto con la lista de features y el horizonte, y `predict` lo levanta de ahí. Reentrenás solo cuando querés incorporar los datos nuevos al modelo.

## Resultados

**Ningún modelo le gana al baseline naive.** El baseline es "predecir siempre la clase mayoritaria", o sea gritar "sube" todos los días. Lo que importa no es el accuracy sino el `edge` = accuracy − baseline:

| Horizonte | Mejor accuracy | Baseline naive | Edge |
|---|---|---|---|
| 1 día | 53.7% | 53.6% | **+0.2%** |
| 5 días | 56.9% | 57.6% | **−0.8%** |
| 10 días | 57.2% | 60.6% | **−3.4%** |
| 21 días | 59.0% | 63.5% | **−4.5%** |

El information coefficient (correlación de rangos entre la confianza del modelo y el retorno real) quedó entre −0.03 y +0.06: ruido.

Vale la pena mirar la trampa que esconde esa tabla. A 21 días el modelo acierta 59%, bastante más que el 53.7% de 1 día — pero el baseline también sube a 63.5%, porque en horizontes largos las acciones suben casi siempre. **Reportar solo el accuracy habría hecho elegir el peor modelo de los cuatro.**

### Limitaciones

- Indicadores técnicos sobre un solo ticker, sin variables macro. Es el punto de partida más simple, y el resultado dice que no alcanza.
- El plan es atacarlo con más información (Fase 5 en `docs/ROADMAP.md`), no tuneando hiperparámetros hasta que el número quede lindo: eso sería sobreajustar la validación y cambiar un modelo honesto por uno que miente mejor.
- El pipeline está completo y validado (sin look-ahead bias, con split temporal); lo que falta es señal.
