"""Tests de integracion: pegan a las APIs reales (yfinance, ArgentinaDatos).

Objetivo: detectar si alguna de las dos fuentes cambia de formato/columnas,
no testear pandas ni la libreria de red. Se aceptan mas lentos que un
unit test comun a cambio de esa cobertura real.
"""

import pytest

import src.fetch
from src.fetch import TICKERS, fetch_ccl, fetch_cedears, fetch_underlying


@pytest.fixture(autouse=True)
def raw_dir_temporal(tmp_path, monkeypatch):
    """Redirige data/raw/ a un directorio temporal.

    Sin esto, el test de abajo (que pide solo desde 2024 para correr rapido)
    pisaba el CSV real del proyecto y dejaba el dataset truncado a 653 filas.
    """
    monkeypatch.setattr(src.fetch, "RAW_DIR", tmp_path)


def test_fetch_underlying_returns_expected_columns():
    df = fetch_underlying("AAPL", start="2024-01-01")

    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert not df.empty
    assert df.index.is_monotonic_increasing


def test_fetch_ccl_returns_expected_columns():
    df = fetch_ccl()

    assert list(df.columns) == ["compra", "venta", "ccl"]
    assert not df.empty
    assert df.index.is_monotonic_increasing


def test_fetch_cedears_returns_expected_columns():
    """data912 es la unica fuente del lado argentino, y de la que mas depende
    `proyeccion.py`: de ahi sale el precio que ancla toda la simulacion.

    Se chequea sobre los tickers de la cartera y no sobre el panel entero: de
    952 simbolos hay decenas sin punta compradora o vendedora (spread negativo o
    sin sentido), que es normal en los CEDEARs que no operan.
    """
    df = fetch_cedears()

    assert list(df.columns) == ["cierre", "bid", "ask", "volumen", "operaciones", "pct_change", "spread"]
    assert set(TICKERS) <= set(df.index)

    cartera = df.loc[TICKERS]
    assert (cartera["cierre"] > 0).all()
    assert (cartera["bid"] <= cartera["ask"]).all()
    assert cartera["spread"].between(0, 0.10).all()
