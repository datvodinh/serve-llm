# Serve LLM with high throughput and scalable using Ray and vLLM

## Development Commands

The Makefile is the primary entry point for local workflows. Run `make help` to see all targets.

```make
LLM Serve - Available targets:

Environment:
  install          Install all dependencies
  lock             Update uv.lock from pyproject
Quality:
  format           Format code via ruff
  lint             Lint via ruff (E,F,S,B)
  fix              Auto-fix (ruff E,F,S,B)
  deptry           Check (un)used deps via deptry
Build:
  build            Build sdist and wheel into dist/
Run:
  run              Run local app (python main.py)
Clean:
  clean            Clean caches, dist, venv (safe)
  clean-all        Deep clean everything
Meta:
  check            format + lint + deptry
  prepare          format + lint + build + clean
```
