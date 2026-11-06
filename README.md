# leakml: PVT-aware CMOS leakage estimation

Course project for estimating transistor, logic-gate, and circuit leakage under process, voltage,
and temperature variation. The implementation builds SPICE data for four transistor operating
cases, trains four current-prediction models, composes the models into five static CMOS gates, and
evaluates the assigned test circuit against a transistor-level SPICE reference.

## Current assignment

| Setting | Value |
|---|---|
| Model card | `models/32nm_LP.pm` |
| Node | 32 nm LP |
| Nominal VDD | 0.9 V |
| Lmin | 32 nm |
| Test circuit | 8-bit odd-parity checker |

The assignment settings and all run knobs live in [`configs/project.yaml`](configs/project.yaml).
The per-person entry points `configs/manya.yaml`, `configs/joanna.yaml`, `configs/sindhuja.yaml`,
and `configs/gopal.yaml` inherit those shared settings and add an `active_person` label. They
intentionally use the same seeds and artifact paths so one owner's stage can consume another
owner's output without copying files. The default remains `configs/project.yaml`.
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

An owner may select their config without changing the shared project parameters, for example
`make train CONFIG=configs/joanna.yaml` or
`uv run python scripts/07_gate_spice.py --config configs/sindhuja.yaml`.

For a small development run using the fixture PDK and stand-in transistor models, see
[`howtorun.md`](howtorun.md). Pilot and development runs write under `_pilot`/`_dev` paths and are
not project deliverables.

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
[`SPEC.md`](SPEC.md) (also linked by [`specs.md`](specs.md)). The module-by-module reference is in
[`docs.md`](docs.md), and execution recipes are in [`howtorun.md`](howtorun.md). Work status and
remaining project deliverables are tracked in [`TODO.md`](TODO.md).

## Project status

The implementation and development checks are in place. The real model environment check and
nominal extraction passed; a one-corner real-card gate smoke check passed all 18 gate/input cases.
The complete 40,000-row assigned transistor data run and the 200-corner gate/circuit evaluation have
not yet been executed, so no assigned-model accuracy claim is reported. See `report/report.md` for
the current setup note and generated-result locations.
