# cedear-ml

Pipeline de ML para predecir el retorno/dirección de CEDEARs. Ver `CLAUDE.md` para las reglas de dominio (por qué se predice retorno y no precio, CCL, look-ahead bias, etc.).

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Uso

```bash
python -m src.fetch          # descarga OHLCV + CCL + macro
python -m src.features       # calcula indicadores tecnicos y target
python -m src.train          # walk-forward validation + entrenamiento
python -m src.predict        # predice con el modelo entrenado
pytest                       # corre los tests
```

Todavía sin implementar más allá del andamiaje — próximo paso: lógica de `src/fetch.py` y `src/features.py`.
