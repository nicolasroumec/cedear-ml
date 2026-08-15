"""Descarga los datos crudos: OHLCV del subyacente y el dolar CCL historico.

No calcula nada (ni indicadores ni target) - eso vive en features.py.
Guarda los resultados en data/raw/ para no tener que volver a pegarle
a las fuentes cada vez que se corre el resto del pipeline.
"""

import io
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# La cartera chica de Fase 5: un CEDEAR liquido por rubro, a proposito de
# rubros poco correlacionados (tech / e-commerce latam / consumo defensivo).
# Apilar tres tech seria triplicar filas sin triplicar informacion: se mueven
# casi juntas, asi que el modelo veria la misma serie tres veces.
TICKERS = ["AAPL", "MELI", "KO"]

CCL_URL = "https://api.argentinadatos.com/v1/cotizaciones/dolares/contadoconliqui"
RIESGO_PAIS_URL = "https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais"

# El endpoint CSV de FRED no pide API key, asi que no hay secreto que manejar.
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

FRED_SERIES = {
    "vix": "VIXCLS",  # volatilidad implicita del S&P 500: el "indice del miedo"
    "yield_curve": "T10Y2Y",  # spread 10 años - 2 años, en negativo anticipa recesion
    "fed_rate": "DFF",  # tasa efectiva de fondos federales
    "cpi": "CPIAUCSL",  # indice de precios al consumidor de EEUU
}

# Cuantos dias tarda cada serie en publicarse desde su fecha de referencia.
# Esto es lo que evita el look-ahead con datos macro: el CPI de junio lleva
# fecha 2026-06-01 pero recien se publica a mediados de julio, asi que usarlo
# en junio seria darle al modelo un dato que en ese momento no existia. Los
# aplica `add_macro()` en features.py.
MACRO_LAGS = {
    "vix": 0,  # cierra junto con la bolsa
    "yield_curve": 0,  # dato de mercado diario
    "fed_rate": 1,  # la tasa efectiva se publica al dia siguiente
    "cpi": 45,  # mensual, sale a mitad del mes siguiente
    "riesgo_pais": 0,  # se publica el mismo dia
}


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


def fetch_macro() -> pd.DataFrame:
    """Descarga las series macro: VIX, curva de tasas, tasa FED, CPI y riesgo pais.

    Las cuatro primeras salen de FRED por su endpoint CSV, que no pide API key.
    El riesgo pais (EMBI Argentina) sale de ArgentinaDatos.

    Guarda las series con su **fecha de referencia**, sin corregir por demora de
    publicacion: esto es data cruda. El desfasaje lo aplica `add_macro()` en
    features.py, asi cambiar el criterio no obliga a volver a descargar.

    Se guarda el historico completo de cada fuente, sin recortar por fecha: el
    CPI interanual necesita 12 meses previos, y recortar en 2018 costaria el
    primer año del dataset. El join contra el subyacente ya hace el recorte.

    Returns:
        DataFrame indexado por fecha con una columna por serie.
    """
    columnas = {}

    for nombre, serie_id in FRED_SERIES.items():
        response = requests.get(FRED_CSV_URL.format(serie_id), timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), parse_dates=["observation_date"])
        # FRED marca los feriados con "." en vez de dejar la fila afuera.
        columnas[nombre] = pd.to_numeric(df.set_index("observation_date")[serie_id], errors="coerce")

    response = requests.get(RIESGO_PAIS_URL, timeout=30)
    response.raise_for_status()
    rp = pd.DataFrame(response.json())
    columnas["riesgo_pais"] = rp.set_index(pd.to_datetime(rp["fecha"]))["valor"]

    macro = pd.DataFrame(columnas).sort_index()
    macro.index.name = "fecha"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    macro.to_csv(RAW_DIR / "macro.csv")
    return macro


if __name__ == "__main__":
    for ticker in TICKERS:
        underlying = fetch_underlying(ticker)
        print(f"{ticker}: {len(underlying)} filas -> {RAW_DIR / f'{ticker.lower()}_ohlcv.csv'}")

    ccl = fetch_ccl()
    macro = fetch_macro()
    print(f"CCL: {len(ccl)} filas -> {RAW_DIR / 'ccl.csv'}")
    print(f"Macro: {len(macro)} filas x {len(macro.columns)} series -> {RAW_DIR / 'macro.csv'}")
