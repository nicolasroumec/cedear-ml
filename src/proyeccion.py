"""Proyeccion a 1-5 anios del valor en pesos de una cartera de CEDEARs.

Esto no usa el modelo de `train.py` y no puede usarlo: a 1-5 anios los
indicadores tecnicos diarios no dicen nada, y en 8 anios de historia entran
~1,6 ventanas independientes de 5 anios. No hay con que entrenar. Lo que si se
puede hacer honestamente es simular.

La identidad de CLAUDE.md es la estructura del problema:

    precio_cedear_ARS = precio_subyacente_USD * CCL / ratio

pero **no se usa para calcular el precio de hoy**, y esa es la decision central
del modulo. El precio de hoy es el que cotiza en BYMA, que baja
`fetch.fetch_cedears()`, y la simulacion lo multiplica por retornos. Tres cosas
se caen solas con eso:

- El ratio deja de importar para el resultado (solo se usa en `chequear_precios`
  como control), asi que un ajuste del banco depositario no rompe la proyeccion.
- El nivel de precio de yfinance deja de importar: solo se usan sus *retornos*.
  Es lo que salva el calculo cuando el CSV descuadra contra el mercado, que es
  exactamente lo que pasa hoy con AAPL y MELI (ver `chequear_precios`).
- Cualquier premio o descuento del CEDEAR contra su valor teorico ya esta
  adentro del precio de mercado, en vez de tener que modelarlo.

Cada pata se trata distinto porque se sabe distinto:

- **Volatilidad del subyacente**: se estima de los datos, con bootstrap por
  bloques de los retornos historicos. Conserva las colas y los clusters de
  volatilidad reales sin suponer ninguna distribucion.
- **Rentabilidad esperada**: NO se estima. AAPL rindio 26.6% anual desde 2018;
  proyectar eso a 5 anios lo pone en 990 USD. Es extrapolar un periodo
  excepcional, ademas del sesgo de haber elegido AAPL en 2026 sabiendo como
  salio. Va como parametro con default conservador.
- **CCL**: tampoco se estima. 67% anual compuesto desde 2018, pero +23% en 2024,
  +28% en 2025 y +3.4% en lo que va de 2026. Proyectar el historico da un CCL de
  22.000 a 5 anios. Es un escenario que pone quien mira, no una prediccion.

Por eso la salida son percentiles bajo escenarios explicitos. Un numero solo
escondria que lo que mas pesa en el resultado en pesos es justo lo que menos se
sabe, y la unica forma de que eso quede a la vista es mostrar el mismo valor en
pesos de hoy al lado.

Sobre la regla "target: retorno o direccion, nunca precio absoluto" de
CLAUDE.md: sigue valiendo y no se rompe aca. Esa regla prohibe *entrenar* un
modelo contra el nivel de precio, porque aprende a copiar el valor de hoy. Aca
no se entrena nada: se simulan retornos y el nivel es la salida, no el target.
"""

import numpy as np
import pandas as pd

from src.features import _to_date_index
from src.fetch import RAW_DIR

# Cuantos CEDEARs equivalen a una accion del subyacente (ratio 20:1 -> 20).
# Solo lo usa `chequear_precios()`: la proyeccion no depende de esto. Lo ajusta
# el banco depositario, asi que verificar en el panel del broker al pegar los
# precios (ahi figura al lado del ticker).
RATIOS = {"AAPL": 20, "MELI": 120, "KO": 5, "GLOB": 18}

# Editar con la cartera propia: {ticker: cantidad de CEDEARs}.
CARTERA = {"AAPL": 10, "MELI": 2, "KO": 25}

DIAS_HABILES = 252
BLOQUE = 21  # un mes de rueda
DRIFT_DEFAULT = 0.08  # nominal en USD, no el historico de estos tickers
ESCENARIOS_DEVAL = {"10%": 0.10, "20%": 0.20, "30%": 0.30}

# Cuanto pueden discrepar los CCL implicitos antes de avisar. Entre CEDEARs
# liquidos el arbitraje los mantiene dentro de un par de puntos.
TOLERANCIA_CCL = 0.04


def _cierres(ticker: str) -> pd.Series:
    """Serie de cierres ajustados en USD que dejo `fetch.py`."""
    ruta = RAW_DIR / f"{ticker.lower()}_ohlcv.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}. Corre `python -m src.fetch`.")
    return _to_date_index(pd.read_csv(ruta, index_col="fecha"))["Close"]


def _ccl_hoy() -> float:
    ruta = RAW_DIR / "ccl.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}. Corre `python -m src.fetch`.")
    return float(pd.read_csv(ruta, index_col="fecha")["ccl"].iloc[-1])


def _precios_ars() -> pd.Series:
    """Precio en pesos de cada CEDEAR, del panel que bajo `fetch.fetch_cedears()`.

    Es el ancla de toda la proyeccion. A diferencia del resto de `data/raw/`,
    esto es una foto y no una serie: el panel de data912 da el estado del
    momento, asi que envejece en horas y no en dias. `antiguedad_precios()` dice
    de cuando es.
    """
    ruta = RAW_DIR / "cedears.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}. Corre `python -m src.fetch`.")
    return pd.read_csv(ruta, index_col="symbol")["cierre"]


def antiguedad_precios() -> pd.Timedelta:
    """Cuanto hace que se bajo el panel de CEDEARs."""
    ruta = RAW_DIR / "cedears.csv"
    return pd.Timestamp.now() - pd.Timestamp.fromtimestamp(ruta.stat().st_mtime)


def chequear_precios(tickers: list[str] | None = None) -> pd.DataFrame:
    """Control de sanidad de los precios pegados, contra el subyacente y el CCL.

    Cada CEDEAR implica un CCL propio: `precio_ars * ratio / precio_usd`. Como
    los tres se arbitran contra el mismo dolar, tienen que dar casi iguales. Si
    uno se despega hay tres causas posibles, en orden de probabilidad: el precio
    pegado quedo viejo, el ratio cambio, o el CSV de yfinance no cuadra con el
    mercado para ese ticker.

    El desvio se mide contra el CCL de `data/raw/ccl.csv` y no contra la mediana
    de los tres: es un ancla independiente. Comparar contra la mediana falla
    justo cuando mas hace falta — si dos de tres tickers estan mal, la mediana se
    va con ellos y el unico sano queda senalado como el outlier.

    Esto no corrige nada ni afecta la proyeccion (que solo usa retornos): avisa.
    Es el unico lugar donde entran `RATIOS` y el nivel de precio de yfinance.

    Returns:
        DataFrame por ticker con el CCL implicito, el de referencia y el desvio
        entre los dos.
    """
    precios = _precios_ars()
    tickers = tickers or list(RATIOS)
    referencia = _ccl_hoy()
    implicito = pd.Series(
        {t: precios[t] * RATIOS[t] / _cierres(t).iloc[-1] for t in tickers},
        name="ccl_implicito",
    )
    return pd.DataFrame(
        {
            "ccl_implicito": implicito,
            "ccl_referencia": referencia,
            "desvio": implicito / referencia - 1,
        }
    )


def simular_multiplo_usd(
    tickers: list[str],
    anios: float,
    drift_anual: float = DRIFT_DEFAULT,
    n_sim: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Simula cuanto se multiplica el subyacente en USD a `anios` vista.

    Devuelve multiplos y no precios a proposito: el nivel de precio de yfinance
    no entra en ningun lado, solo sus retornos. Un CSV desactualizado o mal
    ajustado en el nivel no afecta el resultado.

    Bootstrap por bloques de un mes de rueda: remuestrear dia a dia destruiria
    los clusters de volatilidad (las rachas malas vienen juntas), que es
    justamente lo que define el percentil 10 a varios anios.

    Los bloques se sortean **una sola vez y se aplican a todos los tickers**:
    asi cada camino simulado combina el mismo tramo de calendario para los tres,
    y la correlacion entre ellos sale de los datos sin tener que estimarla.
    Simular cada ticker por separado los volveria independientes y subestimaria
    el riesgo de la cartera (AAPL y MELI correlacionan 0.38 a 5 dias).

    El drift historico se reemplaza por `drift_anual` antes de simular. La vol
    de 8 anios es estimable; la rentabilidad esperada de un activo, no.

    Args:
        tickers: simbolos a simular juntos.
        anios: horizonte. Se redondea al multiplo de 21 dias mas cercano
            (a 5 anios eso es +-2 semanas, irrelevante al lado del resto).
        drift_anual: rentabilidad nominal anual supuesta en USD, la misma para
            todos: 8 anios no alcanzan para decir cual va a rendir mas.
        n_sim: caminos simulados.

    Returns:
        Array (n_sim, len(tickers)) de multiplos. 1.0 = el precio no se movio.
    """
    log_ret = pd.DataFrame({t: np.log1p(_cierres(t).pct_change()) for t in tickers}).dropna()
    ajustado = log_ret - log_ret.mean() + np.log1p(drift_anual) / DIAS_HABILES

    # Suma acumulada: el retorno de un bloque es una resta entre dos puntos, asi
    # que no hace falta materializar los 1260 dias de cada camino.
    acum = np.vstack([np.zeros(len(tickers)), ajustado.to_numpy().cumsum(axis=0)])

    n_bloques = max(1, round(anios * DIAS_HABILES / BLOQUE))
    rng = np.random.default_rng(seed)
    arranques = rng.integers(0, len(log_ret) - BLOQUE, size=(n_sim, n_bloques))
    return np.exp((acum[arranques + BLOQUE] - acum[arranques]).sum(axis=1))


def proyectar(
    cartera: dict[str, float] = CARTERA,
    anios: float = 3,
    deval_anual: float = 0.20,
    drift_anual: float = DRIFT_DEFAULT,
    n_sim: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Precio proyectado de cada CEDEAR y de la cartera entera, en pesos.

    Args:
        cartera: {ticker: cantidad de CEDEARs}.
        anios: horizonte de la proyeccion.
        deval_anual: cuanto sube el CCL por anio en este escenario. Determinista
            a proposito: no es una variable con historia proyectable, es el
            supuesto de quien mira. Ponerle una distribucion inventada le daria
            al numero una precision que no tiene.
        drift_anual: ver `simular_multiplo_usd`.

    Returns:
        DataFrame con una fila por ticker mas 'CARTERA', y columnas:
        'hoy' (precio actual en pesos, el pegado del broker), 'p10'/'p50'/'p90'
        (percentiles a `anios`), 'p50_pesos_hoy' (la mediana descontada por la
        devaluacion supuesta, o sea cuanto de la suba es real y cuanto es el
        dolar) y 'x_veces' (p50 / hoy).
    """
    tickers = list(cartera)
    panel = _precios_ars()
    faltan = set(tickers) - set(panel.index)
    if faltan:
        raise KeyError(f"{sorted(faltan)} no figura(n) en el panel de CEDEARs de BYMA.")

    cantidades = np.array([cartera[t] for t in tickers], dtype=float)
    precios = panel[tickers].to_numpy(dtype=float)

    factor_ccl = (1 + deval_anual) ** anios
    ars = simular_multiplo_usd(tickers, anios, drift_anual, n_sim, seed) * precios * factor_ccl

    hoy = dict(zip(tickers, precios.tolist()))
    simulado = {t: ars[:, i] for i, t in enumerate(tickers)}
    hoy["CARTERA"] = float(precios @ cantidades)
    simulado["CARTERA"] = ars @ cantidades

    return pd.DataFrame(
        {
            "hoy": hoy,
            "p10": {k: np.percentile(v, 10) for k, v in simulado.items()},
            "p50": {k: np.percentile(v, 50) for k, v in simulado.items()},
            "p90": {k: np.percentile(v, 90) for k, v in simulado.items()},
        }
    ).assign(
        p50_pesos_hoy=lambda d: d["p50"] / factor_ccl,
        x_veces=lambda d: d["p50"] / d["hoy"],
    )


if __name__ == "__main__":
    horas = antiguedad_precios() / pd.Timedelta(hours=1)
    print(f"Precios de BYMA bajados hace {horas:.0f}h   |   drift supuesto: {DRIFT_DEFAULT:.0%} anual en USD")
    print(f"Cartera: {', '.join(f'{q} {t}' for t, q in CARTERA.items())}")
    if horas > 24:
        print("[!] El panel de CEDEARs tiene mas de un dia. Corre `python -m src.fetch`.")

    chequeo = chequear_precios(list(CARTERA))
    peor = chequeo["desvio"].abs().max()
    if peor > TOLERANCIA_CCL:
        print(f"\n[!] Hay CEDEARs que no cierran contra el CCL de referencia (hasta {peor:.1%}).")
        print(chequeo.round(0).assign(desvio=chequeo["desvio"].map("{:+.1%}".format)).to_string())
        print("    Revisar, por orden de probabilidad: precios pegados viejos, ratio")
        print("    cambiado, o el CSV de yfinance descuadrado para ese ticker.")
        print("    La proyeccion sigue valiendo: solo usa retornos, no niveles.")

    for anios in (1, 3, 5):
        print(f"\n{'=' * 78}\nA {anios} anio{'s' if anios > 1 else ''}, escenario devaluacion 20%/anio\n{'=' * 78}")
        d = proyectar(CARTERA, anios, deval_anual=0.20)
        print(
            d.assign(
                **{
                    "hoy $": d["hoy"].map("{:,.0f}".format),
                    "p10 $": d["p10"].map("{:,.0f}".format),
                    "p50 $": d["p50"].map("{:,.0f}".format),
                    "p90 $": d["p90"].map("{:,.0f}".format),
                    "p50 en $ de hoy": d["p50_pesos_hoy"].map("{:,.0f}".format),
                    "x": d["x_veces"].map("{:.2f}x".format),
                }
            )[["hoy $", "p10 $", "p50 $", "p90 $", "p50 en $ de hoy", "x"]].to_string()
        )

    print(f"\n{'=' * 78}\nSensibilidad: valor de la CARTERA segun el supuesto de devaluacion\n{'=' * 78}")
    grilla = pd.DataFrame(
        {
            f"{anios}a": {
                f"deval {nombre}/anio": proyectar(CARTERA, anios, deval).loc["CARTERA", "p50"]
                for nombre, deval in ESCENARIOS_DEVAL.items()
            }
            for anios in (1, 3, 5)
        }
    )
    print(grilla.map("{:,.0f}".format).to_string())

    print(
        "\nLeer con cuidado: la banda p10-p90 es solo el riesgo del subyacente en USD.\n"
        "La devaluacion es un supuesto fijo, no una variable simulada: mira la grilla\n"
        "de arriba para ver cuanto del numero en pesos depende de ese supuesto y no\n"
        "de las acciones. La columna 'p50 en $ de hoy' es la parte que es ganancia real."
    )
