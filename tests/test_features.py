import pytest

from src.features import add_technical_indicators, add_target


# ponytail: features.py aun no tiene logica real. Reemplazar este placeholder
# por un test que arme un DataFrame sintetico y verifique que ninguna columna
# en la fila t depende de valores con fecha > t (regla de look-ahead bias de CLAUDE.md).
def test_add_technical_indicators_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        add_technical_indicators(None)


def test_add_target_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        add_target(None)
