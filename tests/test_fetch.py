"""Tests de integracion: pegan a las APIs reales (yfinance, ArgentinaDatos).

Objetivo: detectar si alguna de las dos fuentes cambia de formato/columnas,
no testear pandas ni la libreria de red. Se aceptan mas lentos que un
unit test comun a cambio de esa cobertura real.
"""

import pytest

import src.fetch
from src.fetch import fetch_ccl, fetch_underlying


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
