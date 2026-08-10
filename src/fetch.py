"""Descarga los datos crudos: OHLCV del subyacente y el dolar CCL historico.

No calcula nada (ni indicadores ni target) - eso vive en features.py.
Guarda los resultados en data/raw/ para no tener que volver a pegarle
a las fuentes cada vez que se corre el resto del pipeline.
"""

from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

CCL_URL = "https://api.argentinadatos.com/v1/cotizaciones/dolares/contadoconliqui"


def fetch_underlying(ticker: str, start: str = "2018-01-01") -> pd.DataFrame:
    """Descarga el historico diario OHLCV (ajustado) de un ticker via yfinance.

    Args:
        ticker: simbolo en el mercado de origen, ej. "AAPL".
        start: fecha de inicio "YYYY-MM-DD". yfinance trae hasta hoy.

    Returns:
        DataFrame indexado por fecha con columnas Open/High/Low/Close/Volume
        (Close ya ajustado por splits y dividendos).
    """
    df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "fecha"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / f"{ticker.lower()}_ohlcv.csv")
    return df


def fetch_ccl() -> pd.DataFrame:
    """Descarga la serie historica completa del dolar CCL (ArgentinaDatos).

    Returns:
        DataFrame indexado por fecha con columnas 'compra', 'venta' y 'ccl'
        (promedio de compra/venta, que es lo que se usa como feature).
    """
    response = requests.get(CCL_URL, timeout=30)
    response.raise_for_status()

    df = pd.DataFrame(response.json())
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.set_index("fecha").sort_index()[["compra", "venta"]]
    df["ccl"] = df[["compra", "venta"]].mean(axis=1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / "ccl.csv")
    return df


if __name__ == "__main__":
    underlying = fetch_underlying("AAPL")
    ccl = fetch_ccl()
    print(f"Subyacente: {len(underlying)} filas -> {RAW_DIR / 'aapl_ohlcv.csv'}")
    print(f"CCL: {len(ccl)} filas -> {RAW_DIR / 'ccl.csv'}")
