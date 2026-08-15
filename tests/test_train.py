"""Test del split temporal. Sin red y sin entrenar nada: solo los indices.

Un split mal armado no rompe nada visible — entrena igual y devuelve metricas
lindas. Por eso se testea explicitamente.
"""

import pandas as pd

from src.train import walk_forward_splits

FECHAS = pd.date_range("2020-01-01", periods=1000, freq="B", name="fecha")


def test_walk_forward_nunca_entrena_con_el_futuro():
    """Todo el train tiene que ser anterior a todo el test, con el hueco puesto.

    El hueco (`gap`) importa porque el target de la fila t usa el precio de
    t+N: sin el, las ultimas filas de train ya contienen el resultado de los
    primeros dias de test, y el modelo aprueba el examen porque lo vio antes.
    """
    horizonte = 5
    splits = list(walk_forward_splits(FECHAS, horizon_days=horizonte))

    assert len(splits) == 5
    for idx_train, idx_test in splits:
        assert max(idx_train) < min(idx_test)
        assert min(idx_test) - max(idx_train) > horizonte


def test_walk_forward_avanza_en_el_tiempo():
    """Cada fold entrena con mas historia y testea mas adelante que el anterior."""
    splits = list(walk_forward_splits(FECHAS, horizon_days=1))

    for (train_previo, test_previo), (train_actual, test_actual) in zip(splits, splits[1:]):
        assert len(train_actual) > len(train_previo)
        assert min(test_actual) > min(test_previo)


def test_walk_forward_no_parte_un_dia_entre_train_y_test():
    """Con varios tickers cada fecha se repite: ninguna puede caer de los dos lados.

    Si el corte fuera posicional, el fold partiria un dia al medio y el modelo
    entrenaria con el AAPL del 3 de marzo para ser evaluado sobre el KO del 3
    de marzo — mismo contexto macro, mismo CCL, misma jornada. El hueco del
    `gap` tambien tiene que seguir midiendo dias y no filas.
    """
    horizonte = 5
    tickers = ["AAPL", "MELI", "KO"]
    fechas = pd.Index(FECHAS.repeat(len(tickers)), name="fecha")

    splits = list(walk_forward_splits(fechas, horizon_days=horizonte))

    assert len(splits) == 5
    for idx_train, idx_test in splits:
        dias_train, dias_test = set(fechas[idx_train]), set(fechas[idx_test])
        assert not dias_train & dias_test
        assert max(dias_train) < min(dias_test)
        # El hueco se mide en dias de rueda, no en filas.
        assert len(FECHAS[(FECHAS > max(dias_train)) & (FECHAS < min(dias_test))]) >= horizonte
        # Cada dia entra completo: las tres filas del mismo dia o ninguna.
        assert len(idx_test) == len(dias_test) * len(tickers)
