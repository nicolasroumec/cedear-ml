"""Indicadores tecnicos y construccion del target (retorno/direccion). Todavia sin implementar.

Regla de CLAUDE.md: ningun feature en la fila de fecha t puede usar datos posteriores a t.
"""


def add_technical_indicators(df):
    raise NotImplementedError


def add_target(df, horizon_days: int = 1):
    raise NotImplementedError


if __name__ == "__main__":
    pass
