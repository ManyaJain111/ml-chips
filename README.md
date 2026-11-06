# leakml: PVT-aware CMOS leakage estimation

Course project for estimating transistor, logic-gate, and circuit leakage under process, voltage,
and temperature variation. The implementation builds SPICE data for four transistor operating
cases, trains four current-prediction models, composes the models into five static CMOS gates, and
evaluates the assigned test circuit against a transistor-level SPICE reference.

## Current assignment

| Setting | Value |
|---|---|
| Model card | `pdk/32nm_LP.pm` |
| Node | 32 nm LP |
| Nominal VDD | 0.9 V |
| Lmin | 32 nm |
| Test circuit | 8-bit odd-parity checker |

The assignment settings and all run knobs live in [`configs/project.yaml`](configs/project.yaml).
[`configs/dev.yaml`](configs/dev.yaml) is the same config with a synthetic model card
(`tests/fixtures/fixture_45nm.pm`, `cla3` circuit) and `_dev` output folders, for developing
stages before their real inputs exist. Its outputs are never deliverables.
The supplied model card describes its own nominal supply as 1.0 V; the project config follows the
team assignment of 0.9 V. Nominal process parameters are parsed from the model card and are never
hard-coded in the sampler.

## Quick start

Requirements: Python 3.11 or newer (the lock file uses Python 3.12), [uv](https://docs.astral.sh/uv/),
GNU Make, and `ngspice` on `PATH`.

```bash
uv sync
make test
make lint
make check
```

To run the complete assigned pipeline after the environment check:

```bash
make all
```

Any step can be run alone with a config, e.g.
`uv run python scripts/10_circuit_spice.py --config configs/project.yaml`.

For a small development run using the fixture PDK and stand-in transistor models:

```bash
make bootstrap CONFIG=configs/dev.yaml QUICK=1
```

Pilot (`--quick`) and development runs write under `_pilot`/`_dev` paths and are not project
deliverables. Parallel ngspice processes are run with `OMP_WAIT_POLICY=passive` (set in the
Makefile), which avoids a 30-40x slowdown from busy-waiting OpenMP threads.

## Pipeline

| Step | Command target / script | Main output |
|---|---|---|
| 00 | `make check` / `scripts/00_check_env.py` | `artifacts/env.json` |
| 01 | `make nominals` / `scripts/01_extract_nominals.py` | `artifacts/nominals.json` |
| 02 | `make data` / `scripts/02_generate_transistor_data.py` | Four `data/transistor/*.csv` datasets |
| 03 | `make datacheck` / `scripts/03_data_check.py` | KCL table and transistor trend plots |
| 04 | `make train` / `scripts/04_train_compare.py` | Validation comparison and candidate models |
| 05 | `make finalize` / `scripts/05_finalize_models.py` | Four final models, test metrics, parity plots |
| 06 | `make corners` / `scripts/06_generate_corners.py` | Shared `data/corners.csv` |
| 07–09 | `make partB` | Gate SPICE, gate ML, tables, parity plot |
| 10–12 | `make partC` | Circuit SPICE, hierarchical ML, tables, parity plot |

The full contracts, data schemas, sign conventions, and scoring definitions are in
[`SPEC.md`](SPEC.md); conventions for contributors are in [`AGENTS.md`](AGENTS.md). Work status and
remaining deliverables are tracked in [`TODO.md`](TODO.md).

## Project status

Done and tested (`make test`): the shared foundation, Part A data generation and checks (the four
10,000-row datasets in `data/transistor/` and the Kirchhoff table and trend plots in
`results/`), and the Part C code (circuit definitions, logic simulation, transistor-level
netlists, SPICE reference, hierarchical ML sum, results table and parity plot). Part C has only
been run on the dev config with stand-in gate results.

Not yet produced: the final transistor models (steps 04-05), the Part B gate results (steps
07-09), the shared `data/corners.csv`, and therefore the assigned-card circuit results (steps
10-12). No assigned-model accuracy is reported yet. See `report/report.md` for the report draft
and [`TODO.md`](TODO.md) for open items.
