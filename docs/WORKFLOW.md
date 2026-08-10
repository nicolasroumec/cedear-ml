# Workflow — ramas y commits

Proyecto de una persona: nada de Gitflow (sin `develop`/`release`). `master` siempre queda deployable/funcionando; el trabajo en curso vive en ramas cortas.

## Ramas

- Una rama por incremento del roadmap (ver `docs/ROADMAP.md`), no por tarea diminuta dentro de esa fase.
- Nombre: `fase/<n>-<slug>`, ej. `fase/1-fetch-datos`, `fase/2-features-target`, `fase/3-train-model`.
- Se mergea a `master` cuando la fase compila, pasa los tests y el checklist de esa fase en el roadmap queda tildado.
- Rama borrada después del merge — no quedan ramas viejas colgando en el remoto.

```bash
git checkout -b fase/1-fetch-datos
# ... trabajo ...
git push -u origin fase/1-fetch-datos
# merge a master (PR en GitHub o merge directo)
git branch -d fase/1-fetch-datos
git push origin --delete fase/1-fetch-datos
```

## Commits

Mensaje corto en imperativo, sin prefijos tipo Conventional Commits (`feat:`, `fix:`) — para un repo chico de una persona es ceremonia de más. Un commit por cambio coherente y funcional, no un commit gigante por fase entera.

Ejemplos:

```
Add OHLCV fetch for underlying via yfinance
Add CCL fetch from dolarapi.com
Add technical indicators with pandas-ta
Add look-ahead bias test for features
```

Antes de cada commit: correr `pytest` y confirmar que los módulos tocados importan sin error — no commitear código que rompe lo ya andando.

## Cuándo actualizar el roadmap

Tildar el checkbox en `docs/ROADMAP.md` en el mismo commit (o el siguiente inmediato) que cierra esa tarea — no dejarlo para "después", se desactualiza rápido.
