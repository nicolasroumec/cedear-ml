"""Walk-forward validation, entrenamiento y metricas.

Regla de CLAUDE.md: split temporal (TimeSeriesSplit), nunca random shuffle.

La pregunta que responde este modulo no es "cuanto acierta el modelo" sino
"cuanto acierta el modelo POR ENCIMA de no saber nada". Un clasificador que
grita "sube" siempre acierta ~54% en AAPL, asi que un accuracy de 54% no es
un modelo: es un loro. Por eso cada fold se compara contra ese baseline y lo
que se reporta es la diferencia (el edge).

ponytail: sin StandardScaler. Los arboles parten por umbrales sobre cada
feature por separado, asi que son invariantes a la escala; normalizar no
cambiaria nada. Si algun dia se prueba regresion logistica o SVM, ahi si
hace falta, y con estadisticas calculadas solo sobre el fold de train.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from src.features import FEATURE_COLUMNS, build_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

N_SPLITS = 5
RANDOM_STATE = 42

MODELOS = {
    "random_forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, random_state=RANDOM_STATE
    ),
}


def walk_forward_splits(n_filas: int, horizon_days: int, n_splits: int = N_SPLITS):
    """Genera los cortes temporales train/test, siempre hacia adelante.

    `gap=horizon_days` descarta las filas justo antes del test: el target de
    la fila t usa el precio de t+N, asi que sin ese hueco las ultimas filas
    de train contendrian el resultado de los primeros dias de test.

    Yields:
        Pares (indices_train, indices_test). Cada test es posterior a su train.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon_days)
    yield from tscv.split(range(n_filas))


def evaluar(df: pd.DataFrame, modelo, horizon_days: int) -> dict:
    """Corre walk-forward con un modelo y devuelve sus metricas promedio.

    En cada fold entrena con el pasado, predice el futuro inmediato, y compara
    contra un baseline que siempre responde la clase mayoritaria del train
    (mayoritaria del TRAIN, no del test: mirar el test seria hacer trampa).

    Returns:
        dict con accuracy, baseline, edge, precision, recall, IC y la matriz
        de confusion acumulada de todos los folds.
    """
    X, y = df[FEATURE_COLUMNS], df["target"]
    ret_fwd = df[f"ret_fwd_{horizon_days}"]

    filas = []
    y_real_total, y_pred_total = [], []

    for idx_train, idx_test in walk_forward_splits(len(df), horizon_days):
        modelo.fit(X.iloc[idx_train], y.iloc[idx_train])

        y_real = y.iloc[idx_test]
        y_pred = modelo.predict(X.iloc[idx_test])
        proba = modelo.predict_proba(X.iloc[idx_test])[:, 1]

        clase_mayoritaria = y.iloc[idx_train].mode()[0]

        filas.append(
            {
                "accuracy": accuracy_score(y_real, y_pred),
                "baseline": accuracy_score(y_real, [clase_mayoritaria] * len(y_real)),
                "precision": precision_score(y_real, y_pred, zero_division=0),
                "recall": recall_score(y_real, y_pred, zero_division=0),
                # Information coefficient: correlacion de rangos entre la
                # confianza del modelo y el retorno que realmente ocurrio.
                # Mide si ordena bien los dias, no solo si acierta el signo.
                "ic": pd.Series(proba).corr(
                    pd.Series(ret_fwd.iloc[idx_test].to_numpy()), method="spearman"
                ),
            }
        )
        y_real_total.extend(y_real)
        y_pred_total.extend(y_pred)

    metricas = pd.DataFrame(filas).mean().to_dict()
    metricas["edge"] = metricas["accuracy"] - metricas["baseline"]
    metricas["matriz_confusion"] = confusion_matrix(y_real_total, y_pred_total)
    return metricas


def train_model(ticker: str = "AAPL", horizon_days: int = 5) -> tuple[str, dict]:
    """Compara los modelos con walk-forward, guarda el mejor y devuelve su nombre.

    El mejor se reentrena sobre TODO el historial antes de guardarse: la
    validacion ya cumplio su funcion (estimar el rendimiento), y para predecir
    manana conviene un modelo que vio hasta ayer.
    """
    df = build_dataset(ticker, horizon_days)
    print(f"{len(df)} filas, {len(FEATURE_COLUMNS)} features, horizonte {horizon_days} dias\n")

    resultados = {}
    for nombre, modelo in MODELOS.items():
        m = evaluar(df, modelo, horizon_days)
        resultados[nombre] = m
        print(f"{nombre}")
        print(f"  accuracy  {m['accuracy']:.1%}   (baseline naive {m['baseline']:.1%})")
        print(f"  edge      {m['edge']:+.1%}   <- lo unico que importa")
        print(f"  precision {m['precision']:.1%}    recall {m['recall']:.1%}")
        print(f"  IC        {m['ic']:+.3f}")
        print(f"  matriz de confusion (filas=real, columnas=predicho):\n{m['matriz_confusion']}\n")

    mejor = max(resultados, key=lambda n: resultados[n]["edge"])

    modelo_final = MODELOS[mejor].fit(df[FEATURE_COLUMNS], df["target"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = MODELS_DIR / f"{ticker.lower()}_{mejor}.pkl"
    joblib.dump({"modelo": modelo_final, "features": FEATURE_COLUMNS, "horizon_days": horizon_days}, ruta)

    importancias = pd.Series(modelo_final.feature_importances_, index=FEATURE_COLUMNS)
    print(f"Mejor: {mejor} (edge {resultados[mejor]['edge']:+.1%}) -> {ruta.name}")
    print(f"Features mas usadas:\n{importancias.nlargest(5).to_string()}")

    return mejor, resultados[mejor]


if __name__ == "__main__":
    train_model()
