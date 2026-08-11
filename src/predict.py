"""Carga el modelo entrenado y predice sobre los datos mas recientes.

El pkl que deja `train.py` no es solo el modelo: trae tambien la lista de
features y el horizonte. Esos dos campos no son decoracion. El modelo espera
las columnas por posicion, no por nombre: si se le pasan en otro orden no
falla, predice cualquier cosa. Por eso aca las columnas se toman siempre del
pkl y nunca de lo que traiga el DataFrame.

Aviso sobre lo que significa el numero que imprime: el modelo elegido en Fase 3
tiene edge negativo (-0.8% contra el baseline naive). Esto cierra el pipeline
de punta a punta, que es lo que pide la Fase 4 del roadmap; no es una senal
para operar.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.features import build_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_model(path: str | Path) -> dict:
    """Levanta el bundle guardado por `train.py`.

    Returns:
        dict con 'modelo' (el clasificador entrenado), 'features' (los nombres
        en el orden en que fue entrenado) y 'horizon_days'.
    """
    return joblib.load(path)


def latest_model_path(ticker: str = "AAPL") -> Path:
    """Devuelve el pkl mas reciente del ticker.

    `train.py` guarda solo el mejor modelo de cada corrida, asi que el mas
    nuevo por fecha de modificacion es el ultimo que gano la comparacion.
    """
    pkls = sorted(MODELS_DIR.glob(f"{ticker.lower()}_*.pkl"), key=lambda p: p.stat().st_mtime)
    if not pkls:
        raise FileNotFoundError(f"No hay modelo para {ticker} en {MODELS_DIR}. Corre `python -m src.train`.")
    return pkls[-1]


def predict(model: dict, df: pd.DataFrame) -> pd.Series:
    """Probabilidad de suba para cada fila de `df`.

    Args:
        model: el bundle devuelto por `load_model`.
        df: dataset con al menos las columnas de `model['features']`.

    Returns:
        Serie indexada por fecha con la probabilidad de que el retorno a
        `horizon_days` sea positivo.
    """
    X = df[model["features"]]
    return pd.Series(model["modelo"].predict_proba(X)[:, 1], index=df.index, name="proba_suba")


def predict_latest(ticker: str = "AAPL") -> pd.Series:
    """Predice sobre el ultimo dia con datos disponibles.

    Usa `require_target=False` porque las filas mas recientes no tienen target
    (su futuro todavia no ocurrio) y son justamente las que interesan.
    """
    model = load_model(latest_model_path(ticker))
    df = build_dataset(ticker, model["horizon_days"], require_target=False)
    return predict(model, df)


if __name__ == "__main__":
    TICKER = "AAPL"
    ruta = latest_model_path(TICKER)
    model = load_model(ruta)
    proba = predict_latest(TICKER)

    fecha = proba.index[-1]
    p = proba.iloc[-1]
    atraso = (pd.Timestamp.today().normalize() - fecha).days

    print(f"Modelo: {ruta.name}  (horizonte {model['horizon_days']} dias)")
    print(f"Ultimo dato: {fecha.date()}  ({atraso} dias atras)")
    if atraso > 5:
        print("  ^ los CSV de data/raw/ estan viejos. Corre `python -m src.fetch` para refrescar.")
    print(f"\nProbabilidad de suba a {model['horizon_days']} dias: {p:.1%}")
    print(f"Direccion predicha: {'SUBE' if p > 0.5 else 'BAJA'}")
    print(f"\nUltimos 5 dias:\n{(proba.tail(5) * 100).round(1).to_string()}")
    print("\nRecordatorio: edge -0.8% contra el baseline naive. El pipeline anda, el modelo todavia no.")
