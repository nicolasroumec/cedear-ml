"""Test del split temporal. Sin red y sin entrenar nada: solo los indices.

Un split mal armado no rompe nada visible — entrena igual y devuelve metricas
lindas. Por eso se testea explicitamente.
"""

from src.train import walk_forward_splits


def test_walk_forward_nunca_entrena_con_el_futuro():
    """Todo el train tiene que ser anterior a todo el test, con el hueco puesto.

    El hueco (`gap`) importa porque el target de la fila t usa el precio de
    t+N: sin el, las ultimas filas de train ya contienen el resultado de los
    primeros dias de test, y el modelo aprueba el examen porque lo vio antes.
    """
    horizonte = 5
    splits = list(walk_forward_splits(n_filas=1000, horizon_days=horizonte))

    assert len(splits) == 5
    for idx_train, idx_test in splits:
        assert max(idx_train) < min(idx_test)
        assert min(idx_test) - max(idx_train) > horizonte


def test_walk_forward_avanza_en_el_tiempo():
    """Cada fold entrena con mas historia y testea mas adelante que el anterior."""
    splits = list(walk_forward_splits(n_filas=1000, horizon_days=1))

    for (train_previo, test_previo), (train_actual, test_actual) in zip(splits, splits[1:]):
        assert len(train_actual) > len(train_previo)
        assert min(test_actual) > min(test_previo)
