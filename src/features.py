from pathlib import Path

import pandas as pd

from src.fetch import RAW_DIR, fetch_ccl, fetch_underlying

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def _to_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el indice a fecha sin hora ni timezone.

    yfinance devuelve el indice con timezone de Nueva York y ArgentinaDatos
    fechas planas: sin esto el join por fecha no matchea nunca.
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True).tz_localize(None).normalize()
    out.index.name = "fecha"
    return out


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega indicadores tecnicos del subyacente, todos en forma relativa.

    Args:
        df: OHLCV del subyacente indexado por fecha (columnas Open/High/Low/
            Close/Volume, con Close ya ajustado por splits y dividendos).

    Returns:
        Copia de `df` con las columnas de features agregadas. Las primeras
        filas tienen NaN porque las ventanas todavia no se llenaron.
    """
    out = df.copy()
    close, high, low = out["Close"], out["High"], out["Low"]

    # Retornos simples: la variable mas basica y ya estacionaria.
    out["ret_1"] = close.pct_change()
    out["ret_5"] = close.pct_change(5)

    # Tendencia: cuan lejos esta el precio de su media (en %, no en dolares).
    out["sma_20_ratio"] = close / close.rolling(20).mean() - 1
    out["ema_12_ratio"] = close / close.ewm(span=12, adjust=False).mean() - 1

    # RSI de Wilder: fuerza relativa de las subas contra las bajas (0-100).
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi_14"] = 100 - 100 / (1 + avg_gain / avg_loss)

    # MACD: diferencia entre dos EMAs, normalizada por precio para que sea
    # comparable entre epocas. El histograma es MACD menos su senal.
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_norm"] = macd / close
    out["macd_hist_norm"] = (macd - macd_signal) / close

    # Bollinger %b: 0 = banda inferior, 1 = banda superior. Y el ancho de las
    # bandas, que es una medida de volatilidad reciente.
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    out["bb_pct_b"] = (close - (sma_20 - 2 * std_20)) / (4 * std_20)
    out["bb_width"] = 4 * std_20 / sma_20

    # ATR de Wilder como % del precio: volatilidad real, incluyendo los gaps
    # entre el cierre de un dia y la apertura del siguiente.
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    out["atr_14_pct"] = true_range.ewm(alpha=1 / 14, adjust=False).mean() / close

    # Volumen contra su propia media: 1.0 = dia normal, 3.0 = dia raro.
    out["volume_ratio"] = out["Volume"] / out["Volume"].rolling(20).mean()

    return out


def add_ccl(df: pd.DataFrame, ccl: pd.DataFrame) -> pd.DataFrame:
    """Une la serie del CCL al dataset del subyacente, por fecha.

    El CCL trae filas de fin de semana y feriados (repite el ultimo valor) y
    el subyacente solo dias de rueda en EEUU: el join es left sobre las fechas
    del subyacente, con ffill para los feriados argentinos que caen en dia
    habil de EEUU. El ffill solo propaga hacia adelante, nunca hacia atras,
    asi que no introduce look-ahead.

    Args:
        df: dataset del subyacente indexado por fecha.
        ccl: salida de `fetch.fetch_ccl()`, con la columna 'ccl'.

    Returns:
        Copia de `df` con 'ccl' y sus retornos a 1 y 5 dias.
    """
    out = df.join(ccl[["ccl"]], how="left")
    out["ccl"] = out["ccl"].ffill()
    out["ccl_ret_1"] = out["ccl"].pct_change()
    out["ccl_ret_5"] = out["ccl"].pct_change(5)
    return out


def add_target(df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    """Agrega el retorno futuro a N dias y su etiqueta binaria.

    Esta es la unica funcion del modulo que mira al futuro, y es a proposito:
    el target de la fila t es el retorno entre t y t+N. Las ultimas N filas
    quedan en NaN porque ese futuro todavia no existe.

    Args:
        df: dataset con columna 'Close'.
        horizon_days: cuantos dias de rueda hacia adelante se mide el retorno.

    Returns:
        Copia de `df` con 'ret_fwd_{N}' (retorno futuro) y 'target' (1 si ese
        retorno es positivo, 0 si no).
    """
    out = df.copy()
    ret_fwd = out["Close"].shift(-horizon_days) / out["Close"] - 1
    out[f"ret_fwd_{horizon_days}"] = ret_fwd
    out["target"] = (ret_fwd > 0).astype("float").where(ret_fwd.notna())
    return out


FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "sma_20_ratio",
    "ema_12_ratio",
    "rsi_14",
    "macd_norm",
    "macd_hist_norm",
    "bb_pct_b",
    "bb_width",
    "atr_14_pct",
    "volume_ratio",
    "ccl_ret_1",
    "ccl_ret_5",
]


def build_dataset(ticker: str = "AAPL", horizon_days: int = 1) -> pd.DataFrame:
    """Arma el dataset final listo para entrenar y lo guarda en data/processed/.

    Lee los CSV que dejo `fetch.py`; si no estan, los descarga.
    """
    ohlcv_path = RAW_DIR / f"{ticker.lower()}_ohlcv.csv"
    ccl_path = RAW_DIR / "ccl.csv"

    underlying = (
        pd.read_csv(ohlcv_path, index_col="fecha")
        if ohlcv_path.exists()
        else fetch_underlying(ticker)
    )
    ccl = pd.read_csv(ccl_path, index_col="fecha") if ccl_path.exists() else fetch_ccl()

    df = _to_date_index(underlying)
    df = add_technical_indicators(df)
    df = add_ccl(df, _to_date_index(ccl))
    df = add_target(df, horizon_days)

    # Se tiran las filas incompletas: las primeras (ventanas sin llenar) y las
    # ultimas (todavia sin futuro conocido).
    df = df.dropna(subset=FEATURE_COLUMNS + ["target"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / f"{ticker.lower()}_dataset.csv")
    return df


if __name__ == "__main__":
    dataset = build_dataset()
    print(f"Dataset: {len(dataset)} filas x {len(FEATURE_COLUMNS)} features")
    print(f"Rango: {dataset.index.min().date()} -> {dataset.index.max().date()}")
    print(f"Balance del target: {dataset['target'].mean():.1%} de dias positivos")
    print(f"Guardado en {PROCESSED_DIR / 'aapl_dataset.csv'}")
