"""Tests de inferencia. Sin red: usan un modelo de juguete entrenado al vuelo.

Lo que se testea no es que el modelo acierte (no acierta), sino que las
columnas lleguen en el orden correcto y que inferencia vea los dias que
entrenamiento tira.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.features import FEATURE_COLUMNS, build_dataset
from src.predict import predict


def _modelo_de_juguete(n_filas: int = 200) -> dict:
    """Bundle con la misma forma que el que guarda `train.py`, entrenado con ruido."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n_filas, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = (X["ret_1"] > 0).astype(int)
    modelo = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    return {"modelo": modelo, "features": FEATURE_COLUMNS, "horizon_days": 5}


def test_predict_respeta_el_orden_de_features_del_bundle():
    """Reordenar las columnas del DataFrame no puede cambiar la prediccion.

    sklearn pasa las columnas por posicion: si `predict` tomara el orden del
    DataFrame en vez del que quedo guardado en el pkl, le estaria dando rsi_14
    donde el modelo espera ret_1. No falla, no avisa: devuelve fruta. Este es
    el unico test que lo detecta.
    """
    model = _modelo_de_juguete()
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(20, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)

    desordenado = df[list(reversed(FEATURE_COLUMNS))]

    assert predict(model, df).equals(predict(model, desordenado))


def test_predict_devuelve_probabilidades_indexadas_por_fecha():
    """Una probabilidad por fila, en [0, 1], conservando el indice de entrada."""
    model = _modelo_de_juguete()
    rng = np.random.default_rng(2)
    idx = pd.date_range("2026-01-01", periods=10, name="fecha")
    df = pd.DataFrame(rng.normal(size=(10, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS, index=idx)

    proba = predict(model, df)

    assert len(proba) == len(df)
    assert proba.index.equals(idx)
    assert proba.between(0, 1).all()


def test_inferencia_ve_los_dias_que_entrenamiento_descarta():
    """`require_target=False` tiene que devolver las ultimas N filas de mas.

    Son las que todavia no tienen futuro conocido — o sea, las unicas sobre
    las que tiene sentido predecir. Si este test falla, `predict_latest` esta
    prediciendo sobre un dia que ya paso.
    """
    horizonte = 5
    con_target = build_dataset("AAPL", horizonte, require_target=True)
    sin_target = build_dataset("AAPL", horizonte, require_target=False)

    assert len(sin_target) == len(con_target) + horizonte
    assert sin_target.index[-1] > con_target.index[-1]
    assert sin_target[FEATURE_COLUMNS].notna().all().all()


def test_inferencia_no_pisa_el_dataset_de_entrenamiento(tmp_path, monkeypatch):
    """Correr inferencia no puede modificar data/processed/.

    Mismo bug que ya mordio una vez con test_fetch pisando data/raw/: el CSV
    queda con filas sin target y el proximo entrenamiento que lo lea arranca
    con datos distintos sin que nadie se entere.
    """
    import src.features

    monkeypatch.setattr(src.features, "PROCESSED_DIR", tmp_path)
    build_dataset("AAPL", 5, require_target=False)

    assert list(tmp_path.iterdir()) == []
