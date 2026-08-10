"""Tests de integracion: pegan a las APIs reales (yfinance, ArgentinaDatos).

Objetivo: detectar si alguna de las dos fuentes cambia de formato/columnas,
no testear pandas ni la libreria de red. Se aceptan mas lentos que un
unit test comun a cambio de esa cobertura real.
"""

from src.fetch import fetch_ccl, fetch_underlying


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
