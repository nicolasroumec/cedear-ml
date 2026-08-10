# cedear-ml

Pipeline de ML para predecir el retorno/dirección de CEDEARs (no el precio absoluto). Estado actual: solo documentación de referencia, sin código todavía.

Guía completa para humanos (mecánica del instrumento, indicadores, fuentes, trampas de ML): https://claude.ai/code/artifact/dbcddce8-fd01-4ef2-ac05-e27aea564962

## Reglas de dominio que no son negociables

- **Precio del CEDEAR = precio_subyacente_USD × ratio × CCL.** Son dos series independientes (mercado de EEUU + tipo de cambio implícito argentino). No tratar el precio en pesos como una sola señal.
- **CCL** = precio en pesos de una acción dual-listada (ej. GGAL) / precio en USD ajustado por ratio. Se necesita como serie propia, no derivarlo solo del CEDEAR que se está prediciendo.
- **Target: retorno o dirección, nunca precio absoluto.** Un modelo entrenado contra el nivel de precio aprende a copiar el valor de hoy (el precio es un random walk aproximado) y da métricas engañosamente buenas sin ser útil.
- **Usar adjusted close** del subyacente, no el crudo — splits/dividendos generan saltos falsos si no.
- **Sin look-ahead bias:** ningún feature en la fila de fecha `t` puede depender de datos posteriores a `t`. Incluye medias/std usadas para normalizar.
- **Split temporal (walk-forward), nunca random shuffle** entre train/val — mezclar al azar filtra el futuro al pasado.
- **Normalización:** estadísticas (media, desvío) calculadas solo sobre el set de entrenamiento, aplicadas después al de validación.
- **Ratio de conversión del CEDEAR cambia** por ajustes del banco depositario — verificar el vigente antes de construir series largas; un cambio de ratio no ajustado se ve como un salto de precio que no es señal de mercado.

## Fuentes de datos previstas

| Dato | Fuente |
|---|---|
| OHLCV subyacente, índices (S&P 500) | `yfinance` |
| Tasa FED, CPI EEUU | FRED |
| Tasa/oficial Argentina | BCRA (API pública) |
| CCL histórico | dolarapi.com / Bluelytics |
| Cotización CEDEAR en pesos, ratio vigente | IOL / Rava / ByMA Data |

## Convenciones de código

Todavía no definidas — no hay código en el repo. Actualizar esta sección cuando se fije el lenguaje/estructura de carpetas en vez de anticiparla acá.
