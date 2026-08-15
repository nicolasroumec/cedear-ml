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
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from src.features import FEATURE_COLUMNS, build_multi_dataset
from src.fetch import TICKERS

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


def walk_forward_splits(fechas, horizon_days: int, n_splits: int = N_SPLITS):
    """Genera los cortes temporales train/test, siempre hacia adelante.

    El corte es por **fecha**, no por posicion de fila. Con varios tickers
    apilados cada fecha aparece una vez por ticker, y cortar por posicion
    partiria un mismo dia entre train y test — el modelo entrenaria con el
    AAPL del 3 de marzo para despues ser evaluado sobre el KO del 3 de marzo,
    que comparte todo el contexto macro. Ademas asi el `gap` sigue midiendo
    dias de rueda: con 3 tickers, 5 filas son menos de dos dias.

    `gap=horizon_days` descarta los dias justo antes del test: el target del
    dia t usa el precio de t+N, asi que sin ese hueco los ultimos dias de
    train contendrian el resultado de los primeros dias de test.

    Args:
        fechas: el indice de fechas del dataset, con repetidos si hay varios
            tickers.
        horizon_days: horizonte del target, en dias de rueda.

    Yields:
        Pares (posiciones_train, posiciones_test), posicionales sobre `fechas`.
        Cada test es posterior a su train y ninguna fecha cae en los dos.
    """
    fechas = pd.Index(fechas)
    dias = fechas.unique().sort_values()
    posiciones = np.arange(len(fechas))

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon_days)
    for i_train, i_test in tscv.split(dias):
        yield (
            posiciones[fechas.isin(dias[i_train])],
            posiciones[fechas.isin(dias[i_test])],
        )


def evaluar(df: pd.DataFrame, modelo, horizon_days: int) -> dict:
    """Corre walk-forward con un modelo y devuelve sus metricas promedio.

    En cada fold entrena con el pasado, predice el futuro inmediato, y compara
    contra un baseline que siempre responde la clase mayoritaria del train
    (mayoritaria del TRAIN, no del test: mirar el test seria hacer trampa).

    El baseline se calcula **por ticker**: KO no sube el mismo porcentaje de
    dias que MELI, asi que un unico baseline sobre el pool le regalaria edge al
    modelo en los tickers mas alcistas y se lo cobraria en los otros.

    Returns:
        dict con accuracy, baseline, edge, precision, recall, IC, la matriz de
        confusion acumulada y `por_ticker`, el mismo edge abierto por activo.
    """
    X, y, tickers = df[FEATURE_COLUMNS], df["target"], df["ticker"]
    ret_fwd = df[f"ret_fwd_{horizon_days}"]

    filas = []
    predicciones = []

    for idx_train, idx_test in walk_forward_splits(df.index, horizon_days):
        modelo.fit(X.iloc[idx_train], y.iloc[idx_train])

        y_real = y.iloc[idx_test]
        y_pred = modelo.predict(X.iloc[idx_test])
        proba = modelo.predict_proba(X.iloc[idx_test])[:, 1]

        mayoritaria = y.iloc[idx_train].groupby(tickers.iloc[idx_train]).agg(lambda s: s.mode()[0])
        y_base = tickers.iloc[idx_test].map(mayoritaria).to_numpy()

        filas.append(
            {
                "accuracy": accuracy_score(y_real, y_pred),
                "baseline": accuracy_score(y_real, y_base),
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
        predicciones.append(
            pd.DataFrame(
                {"ticker": tickers.iloc[idx_test].to_numpy(), "real": y_real.to_numpy(),
                 "pred": y_pred, "base": y_base}
            )
        )

    metricas = pd.DataFrame(filas).mean().to_dict()
    metricas["edge"] = metricas["accuracy"] - metricas["baseline"]

    todas = pd.concat(predicciones, ignore_index=True)
    metricas["matriz_confusion"] = confusion_matrix(todas["real"], todas["pred"])
    metricas["por_ticker"] = todas.groupby("ticker").apply(
        lambda g: pd.Series(
            {
                "accuracy": (g["real"] == g["pred"]).mean(),
                "baseline": (g["real"] == g["base"]).mean(),
                "n": len(g),
            }
        ),
        include_groups=False,
    ).assign(edge=lambda d: d["accuracy"] - d["baseline"])
    return metricas


def train_model(tickers: list[str] = TICKERS, horizon_days: int = 5) -> tuple[str, dict]:
    """Compara los modelos con walk-forward, guarda el mejor y devuelve su nombre.

    Entrena un solo modelo sobre los tickers apilados, no uno por ticker: la
    apuesta de Fase 5 es que el patron a aprender es comun a todos y lo que
    faltaba eran filas. Si fuera al reves —cada activo con su propia logica—
    se veria como un modelo pooled peor que los individuales.

    El mejor se reentrena sobre TODO el historial antes de guardarse: la
    validacion ya cumplio su funcion (estimar el rendimiento), y para predecir
    manana conviene un modelo que vio hasta ayer.
    """
    df = build_multi_dataset(tickers, horizon_days)
    print(f"{len(df)} filas ({', '.join(tickers)}), {len(FEATURE_COLUMNS)} features, "
          f"horizonte {horizon_days} dias\n")

    resultados = {}
    for nombre, modelo in MODELOS.items():
        m = evaluar(df, modelo, horizon_days)
        resultados[nombre] = m
        print(f"{nombre}")
        print(f"  accuracy  {m['accuracy']:.1%}   (baseline naive {m['baseline']:.1%})")
        print(f"  edge      {m['edge']:+.1%}   <- lo unico que importa")
        print(f"  precision {m['precision']:.1%}    recall {m['recall']:.1%}")
        print(f"  IC        {m['ic']:+.3f}")
        print(f"  por ticker (accuracy / baseline / edge):")
        print(f"{(m['por_ticker'][['accuracy', 'baseline', 'edge']] * 100).round(1).to_string()}")
        print(f"  matriz de confusion (filas=real, columnas=predicho):\n{m['matriz_confusion']}\n")

    mejor = max(resultados, key=lambda n: resultados[n]["edge"])

    modelo_final = MODELOS[mejor].fit(df[FEATURE_COLUMNS], df["target"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = MODELS_DIR / f"cartera_{mejor}.pkl"
    joblib.dump(
        {
            "modelo": modelo_final,
            "features": FEATURE_COLUMNS,
            "horizon_days": horizon_days,
            "tickers": tickers,
        },
        ruta,
    )

    importancias = pd.Series(modelo_final.feature_importances_, index=FEATURE_COLUMNS)
    print(f"Mejor: {mejor} (edge {resultados[mejor]['edge']:+.1%}) -> {ruta.name}")
    print(f"Features mas usadas:\n{importancias.nlargest(5).to_string()}")

    return mejor, resultados[mejor]


if __name__ == "__main__":
    train_model()
