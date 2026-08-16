"""Tests de la proyeccion. Sin red y sin leer data/raw/: se falsean las dos
funciones que tocan disco (`_cierres` y `_ccl_hoy`) y todo lo demas es calculo.
"""

import numpy as np
import pandas as pd
import pytest

from src import proyeccion


def _serie(seed: int = 0, n: int = 2000) -> pd.Series:
    """Random walk deterministico, sin drift, con vol tipica de una accion."""
    rng = np.random.default_rng(seed)
    return pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
        index=pd.date_range("2018-01-01", periods=n, freq="B", name="fecha"),
    )


def _panel(**precios) -> pd.Series:
    """Panel de CEDEARs con la forma que devuelve `fetch.fetch_cedears()`."""
    return pd.Series(precios or {"AAA": 20_000.0, "BBBB": 30_000.0}, name="cierre")


@pytest.fixture(autouse=True)
def datos_falsos(monkeypatch):
    monkeypatch.setattr(proyeccion, "_cierres", lambda t: _serie(seed=len(t)))
    monkeypatch.setattr(proyeccion, "_ccl_hoy", lambda: 1500.0)
    monkeypatch.setattr(proyeccion, "_precios_ars", _panel)
    monkeypatch.setattr(proyeccion, "RATIOS", {"AAA": 10, "BBBB": 10})


def test_el_precio_de_hoy_es_el_que_cotiza_en_byma():
    """No se recalcula con ratio ni CCL: el ancla es el precio de mercado.

    Si esto falla, la proyeccion volvio a depender del nivel de yfinance y del
    ratio — que es justo lo que descuadro contra el broker en AAPL y MELI.
    """
    d = proyeccion.proyectar({"AAA": 3}, anios=3, deval_anual=0.20)

    assert d.loc["AAA", "hoy"] == 20_000
    assert d.loc["CARTERA", "hoy"] == 60_000


def test_el_nivel_de_yfinance_no_afecta_el_resultado(monkeypatch):
    """Multiplicar por 10 la serie del subyacente no cambia nada.

    Solo se usan retornos. Es la propiedad que hace que un CSV desactualizado en
    el nivel siga sirviendo para proyectar.
    """
    antes = proyeccion.proyectar({"AAA": 1}, anios=3, deval_anual=0.20)
    monkeypatch.setattr(proyeccion, "_cierres", lambda t: _serie(seed=len(t)) * 10)
    despues = proyeccion.proyectar({"AAA": 1}, anios=3, deval_anual=0.20)

    pd.testing.assert_frame_equal(antes, despues)


def test_sin_drift_ni_devaluacion_la_mediana_se_queda_donde_esta():
    """Con drift 0% y CCL quieto, la mediana no puede alejarse del precio de hoy.

    Es el chequeo de que el recentrado del drift funciona: los retornos
    historicos se demedian antes de simular, asi que la suma de un camino tiene
    mediana ~0 y el precio proyectado ~el actual. Si el recentrado estuviera mal
    (o no estuviera), la mediana arrastraria el drift historico de la serie.
    """
    d = proyeccion.proyectar({"AAA": 1}, anios=3, deval_anual=0.0, drift_anual=0.0)

    assert d.loc["AAA", "p50"] == pytest.approx(d.loc["AAA", "hoy"], rel=0.10)


def test_el_drift_supuesto_manda_sobre_el_historico():
    """Subir el drift tiene que subir la mediana en la proporcion esperada."""
    base = proyeccion.proyectar({"AAA": 1}, anios=5, deval_anual=0.0, drift_anual=0.0)
    con_drift = proyeccion.proyectar({"AAA": 1}, anios=5, deval_anual=0.0, drift_anual=0.10)

    assert con_drift.loc["AAA", "p50"] == pytest.approx(base.loc["AAA", "p50"] * 1.10**5, rel=0.02)


def test_percentiles_ordenados_y_cartera_consistente():
    d = proyeccion.proyectar({"AAA": 2, "BBBB": 3}, anios=3, deval_anual=0.20)

    assert (d["p10"] < d["p50"]).all() and (d["p50"] < d["p90"]).all()
    assert d.loc["CARTERA", "hoy"] == 2 * 20_000 + 3 * 30_000


def test_la_devaluacion_escala_el_precio_en_pesos_exactamente():
    """El CCL entra como factor determinista: duplicar el factor duplica la salida.

    Y 'p50_pesos_hoy' tiene que quedar igual, porque es justamente la mediana
    limpia del efecto devaluacion.
    """
    sin = proyeccion.proyectar({"AAA": 1}, anios=2, deval_anual=0.0)
    con = proyeccion.proyectar({"AAA": 1}, anios=2, deval_anual=0.20)

    assert con.loc["AAA", "p50"] == pytest.approx(sin.loc["AAA", "p50"] * 1.20**2)
    assert con.loc["AAA", "p50_pesos_hoy"] == pytest.approx(sin.loc["AAA", "p50_pesos_hoy"])


def test_los_tickers_comparten_los_mismos_bloques(monkeypatch):
    """Dos series identicas tienen que dar caminos identicos.

    Es el test de la unica decision no obvia del modulo: los bloques se sortean
    una vez para todos los tickers, no uno por ticker. Si cada uno tomara los
    suyos, dos series iguales darian caminos distintos — y la cartera quedaria
    diversificada por un artefacto de la simulacion, subestimando su riesgo.
    """
    monkeypatch.setattr(proyeccion, "_cierres", lambda t: _serie(seed=0))

    multiplos = proyeccion.simular_multiplo_usd(["AAA", "BBBB"], anios=3, n_sim=500)

    assert np.allclose(multiplos[:, 0], multiplos[:, 1])


def test_chequear_precios_detecta_un_ratio_cambiado(monkeypatch):
    """Es el control que encontro que MELI era 120:1 y no 30:1.

    Con los ratios bien, los tres CEDEARs implican el mismo CCL. Con uno mal, ese
    ticker se despega y el desvio lo delata.
    """
    monkeypatch.setattr(proyeccion, "_cierres", lambda t: _serie(seed=0))
    # Precios que implican exactamente el CCL de referencia (1500) con ratio 10.
    coherente = 1500 * _serie(seed=0).iloc[-1] / 10
    monkeypatch.setattr(proyeccion, "_precios_ars", lambda: _panel(AAA=coherente, BBBB=coherente))

    ok = proyeccion.chequear_precios(["AAA", "BBBB"])
    assert (ok["desvio"].abs() < proyeccion.TOLERANCIA_CCL).all()

    monkeypatch.setattr(proyeccion, "RATIOS", {"AAA": 10, "BBBB": 40})
    mal = proyeccion.chequear_precios(["AAA", "BBBB"])
    assert (mal["desvio"].abs() > proyeccion.TOLERANCIA_CCL).any()


def test_reproducible():
    a = proyeccion.proyectar({"AAA": 1}, anios=3, deval_anual=0.20, seed=7)
    b = proyeccion.proyectar({"AAA": 1}, anios=3, deval_anual=0.20, seed=7)

    pd.testing.assert_frame_equal(a, b)
