"""Tests de features: el que importa es el de look-ahead bias.

A diferencia de test_fetch.py, estos no tocan la red: arman un DataFrame
sintetico y verifican propiedades del calculo.
"""

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, add_ccl, add_target, add_technical_indicators


def _synthetic_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Serie de precios pseudo-aleatoria pero deterministica, con OHLC coherente."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.005, n)),
            "High": close * (1 + abs(rng.normal(0, 0.01, n))),
            "Low": close * (1 - abs(rng.normal(0, 0.01, n))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="B", name="fecha"),
    )


def _synthetic_ccl(index: pd.DatetimeIndex, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    valores = 500 * np.exp(np.cumsum(abs(rng.normal(0.002, 0.01, len(index)))))
    return pd.DataFrame({"ccl": valores}, index=index)


def test_features_no_look_ahead():
    """Ningun feature de la fila t puede cambiar si se borra todo lo posterior a t.

    Este es el test central de CLAUDE.md: se calculan los features sobre la
    serie completa y sobre la serie truncada en t, y los valores de la fila t
    tienen que ser identicos. Si algun indicador usara datos futuros (por
    ejemplo un rolling centrado, o normalizar con la media de toda la serie),
    los dos calculos difieren y esto falla.
    """
    ohlcv = _synthetic_ohlcv()
    ccl = _synthetic_ccl(ohlcv.index)

    completo = add_ccl(add_technical_indicators(ohlcv), ccl)

    for corte in (60, 120, 199):
        truncado = add_ccl(add_technical_indicators(ohlcv.iloc[: corte + 1]), ccl)
        pd.testing.assert_series_equal(
            completo[FEATURE_COLUMNS].iloc[corte],
            truncado[FEATURE_COLUMNS].iloc[corte],
        )


def test_target_mira_al_futuro_y_no_al_pasado():
    """El target si tiene que mirar adelante: es lo que el modelo debe predecir.

    Contracara del test anterior: verifica que 'target' de la fila t sea el
    signo del retorno entre t y t+N, y que las ultimas N filas queden en NaN
    porque ese futuro todavia no ocurrio.
    """
    ohlcv = _synthetic_ohlcv(n=50)
    horizonte = 3
    df = add_target(ohlcv, horizon_days=horizonte)

    esperado = ohlcv["Close"].iloc[10 + horizonte] / ohlcv["Close"].iloc[10] - 1
    assert df[f"ret_fwd_{horizonte}"].iloc[10] == esperado
    assert df["target"].iloc[10] == float(esperado > 0)
    assert df["target"].iloc[-horizonte:].isna().all()


def test_ccl_se_propaga_hacia_adelante_nunca_hacia_atras():
    """Un feriado argentino toma el CCL del dia anterior, no el del dia siguiente."""
    ohlcv = _synthetic_ohlcv(n=30)
    ccl = _synthetic_ccl(ohlcv.index)
    feriado = ohlcv.index[10]
    ccl_con_hueco = ccl.drop(index=feriado)

    unido = add_ccl(ohlcv, ccl_con_hueco)

    assert unido["ccl"].loc[feriado] == ccl["ccl"].iloc[9]
    assert unido["ccl"].loc[feriado] != ccl["ccl"].iloc[11]


def test_indicadores_en_rangos_esperados():
    """Chequeo barato de sanidad: RSI en 0-100 y ATR/volumen positivos."""
    df = add_technical_indicators(_synthetic_ohlcv())

    assert df["rsi_14"].dropna().between(0, 100).all()
    assert (df["atr_14_pct"].dropna() > 0).all()
    assert (df["volume_ratio"].dropna() > 0).all()
