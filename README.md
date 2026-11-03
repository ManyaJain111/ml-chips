# leakml: ML-driven PVT-variation-aware leakage estimation

Digital VLSI Design course project. Brief: [DVD_ML_PROJECT.pdf](DVD_ML_PROJECT.pdf).

- [SPEC.md](SPEC.md): technical spec and module contracts
- [TODO.md](TODO.md): 4-person work split and milestones
- [AGENTS.md](AGENTS.md): rules and conventions for humans and AI agents

## Quickstart
```bash
uv sync
make test
# Develop now, on a synthetic model card, with stand-in inputs for every stage (~20 s):
make bootstrap CONFIG=configs/dev.yaml QUICK=1

# Real run: fill configs/project.yaml with the TA assignment and drop the PTM file into pdk/
make check            # ngspice + model file + versions -> artifacts/env.json
make partA QUICK=1    # pilot run
make all              # full pipeline (scripts 00 -> 12)
```
See [TODO.md](TODO.md) → "Day 1" and SPEC §11 for how the four parts are developed in parallel.

Reproducible container (pinned ngspice + Python deps):
```bash
make docker-build
make docker-run T=all
```

## Pipeline
| Step | Script | Produces |
|---|---|---|
| 00 | check_env | `artifacts/env.json` |
| 01 | extract_nominals | `artifacts/nominals.json` |
| 02 | generate_transistor_data | `data/transistor/*.csv` (D1) |
| 03 | data_check | Kirchhoff table + trend plots (D3) |
| 04 | train_compare | model comparison table (D4) |
| 05 | finalize_models | `artifacts/models/*.joblib`, test metrics (D5) |
| 06 | generate_corners | `data/corners.csv` |
| 07 | gate_spice | `data/gates/gate_spice.csv` |
| 08 | gate_ml | `data/gates/gate_ml.csv` |
| 09 | gate_report | gate table + parity plot (D6, D7) |
| 10 | circuit_spice | `data/circuit/circuit_spice.csv` |
| 11 | circuit_ml | `data/circuit/circuit_ml.csv` |
| 12 | circuit_report | circuit table + parity plot (D8) |
