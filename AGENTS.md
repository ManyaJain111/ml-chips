# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Copilot, Cursor, …) **and humans** working
in this repo. Read this, then [SPEC.md](SPEC.md) (contracts) and [TODO.md](TODO.md) (who owns what).

## What this is
Course project: predict CMOS leakage under PVT variation with ML trained on single-transistor
ngspice simulations, then validate at gate level (INV/NAND2/NOR2/AND2/OR2) and on a test circuit
(3-bit CLA or 8-bit parity). Brief: `DVD_ML_PROJECT.pdf`.

## Commands
```bash
uv sync                          # install (Python 3.12, deps from uv.lock)
make test                        # pytest (real-ngspice tests skip if ngspice is missing)
make lint / make fmt             # ruff
make bootstrap CONFIG=configs/dev.yaml QUICK=1   # dev inputs for every stage (~20 s), see SPEC §11
make check                       # env sanity check (ngspice, model file, versions)
make partA QUICK=1               # pilot run, small samples, writes to *_pilot folders
make all                         # full pipeline, scripts 00 -> 12 in order
make docker-build && make docker-run T=all   # same, inside the pinned container
uv run python scripts/07_gate_spice.py --config configs/dev.yaml --quick   # a single step
```
Until the assigned PDK is in `pdk/`, use `--config configs/dev.yaml` (synthetic card). Code never
changes between dev and real configs.

## Layout
```
configs/project.yaml   assignment + every knob (seeds, sample counts, sizing, paths)
configs/dev.yaml       same knobs, synthetic model card, _dev output folders (never deliverables)
src/leakml/            library: constants, config, io, pdk, sampling, spice/, data/, ml/, gates/,
                       circuit/, metrics, plotting, cli
scripts/NN_*.py        thin pipeline steps. Logic belongs in src/, not here
scripts/dev_*.py       stand-in generators for independent development (SPEC §11)
tests/                 pytest. Placeholder tests are skip-marked until their module is implemented
tests/fixtures/        synthetic BSIM4 card used by tests and configs/dev.yaml
data/ artifacts/ results/ netlists/examples/   generated deliverables (tracked in git)
report/                the written report
```

## Hard rules
1. **No leakage of evaluation data into training.** Nothing under `data/gates/` or `data/circuit/`
   may be used to train, tune or select transistor models. `gates/ml_model.py` may read only the
   **node voltages** from `gate_spice.csv`, never its currents.
2. **The test split is touched once**, in `scripts/05_finalize_models.py`. Model selection and
   tuning use validation only.
3. **All randomness is seeded** through `sampling.make_rng(cfg.seeds.<x>, label)`. No bare
   `np.random.*`, `random.*`, or estimators without `random_state`.
4. **Every SPICE deck contains `constants.SPICE_OPTIONS` verbatim.** Single-FET decks have one
   voltage source per terminal.
5. **Sign convention:** stored currents are positive INTO the device terminal (`id = -i(vd)` in
   ngspice). Leakage = current delivered by the VDD supply (SPEC §3.5).
6. **SI units:** V, A, metres (W/L), °C. No nm or nA in stored data.
7. **No hard-coded assignment values.** Paths, VDD, Lmin, seeds and counts come from
   `load_config()`.
8. **Contracts live in `SPEC.md` + `constants.py`.** To change a column, signature or file path:
   update SPEC.md first, then the code on both sides, and mention it in the PR.
9. **`data/corners.csv` is frozen** once Part B/C SPICE results exist. Changing it means re-running
   steps 07–12.
10. **`--quick` must never overwrite real deliverables.** It writes to the `_pilot` paths.
11. **Gate topologies come from `gates/topology.py` only.** The gate decks, the ML
    terminal-voltage mapping and the circuit expansion all derive from it.
12. **CSV I/O goes through `leakml.io`** (`read_csv`, `write_csv`, `save_table`). Plain
    `pd.read_csv` turns `"01"` into `1` and corrupts gate/vector keys.
13. **Stand-ins never reach deliverable folders.** `ml/baseline.py` models and placeholder
    `gate_ml.csv` exist only under `_dev*` / `_pilot` paths (the dev scripts enforce this).

## Ownership
| Path | Owner |
|---|---|
| `pdk.py`, `sampling.py`, `spice/`, `data/`, `plotting.py`, scripts 01–03 | P1 (Manya) |
| `ml/`, `metrics.py`, scripts 04–05 | P2 (Joanna) |
| `gates/`, scripts 06–09 | P3 (Sindhuja) |
| `config.py`, `circuit/`, scripts 00 and 10–12, Makefile, Dockerfile, `report/` | P4 (Gopal) |
| `constants.py`, `io.py`, `cli.py`, SPEC.md, `scripts/dev_*` | shared, any change needs agreement |

**Already implemented (the foundation):** everything other stages call into, i.e. config, io,
pdk, sampling, spice/, data/generate, metrics, plotting, ml/predictor (+ the baseline stand-in),
gates/topology, `render_gate_subckt`, `GateLeakageModel`, and scripts 00/01/02/06. These are
covered by tests. The owner fixes bugs in place but must not change their signatures (hard
rule 8). TODO.md lists what is still open per person.

**Agents:** edit only files owned by the person you are working for, unless explicitly asked. If a
fix belongs in someone else's module, say so (or leave a `TODO(Px):` note) instead of silently
changing it.

## Implementing a stub
Every unimplemented function has a docstring with a `TODO(Px):` block describing what to do, and
ends in `raise NotImplementedError`. To implement one:
1. Read the docstring and the SPEC section it cites.
2. Keep the signature (it is a contract). If it must change, follow hard rule 8.
3. Replace the `raise NotImplementedError`. Keep the useful parts of the docstring and delete
   finished TODO lines.
4. Remove the `pytestmark = pytest.mark.skip(...)` from the matching test file and fill in the tests.
5. `make test lint` must be green. Tick the box in TODO.md.

## Code conventions
- Python ≥ 3.11 (pinned to 3.12 via `.python-version`), full type hints, `ruff` clean (line length 100).
- DataFrames carry the column names defined in SPEC/`constants.py`. Don't invent ad-hoc names.
- Vectorise. ML prediction happens in **one batched call** per stage, not per row.
- Library code logs with `logging`. Scripts may `print` summaries.
- Write CSVs with `io.write_csv` (full precision, `%.10e`). Leakage currents are tiny.
- Figures go through `plotting.save_fig` (PNG + PDF, Agg backend).

## Domain pitfalls (read before touching SPICE or ML code)
- PTM PMOS `vth0` is **negative**. ±10 % scales the value and keeps the sign.
- ngspice's default print precision is too low: always `set numdgt=12`.
- `i(Vx)` in ngspice flows + → − through the source. Getting the sign wrong breaks KCL, and the
  Kirchhoff check will catch it.
- Probing nodes inside a subckt: `v(x1.x)` (verified on ngspice 47, used by `test_gate_subckt_logic`).
- `i(vvdd)` is negative when the supply delivers current: `i_vdd_spice = -i(vvdd)`.
- The Makefile is `.NOTPARALLEL`: steps consume each other's outputs, so don't run them with `-j`.
- Currents span ~10 decades and change sign. Regress in log/asinh space, and judge models by
  relative error, not MSE in amperes.
- Relative error explodes where a current crosses zero (`id` when `vd ≈ vs`). Use the `i_floor`
  rule (SPEC §7).
- The transistor models only saw `vg ∈ {0, VDD}`. In AND2/OR2 the INV stage's gate (`n1`) is a
  recorded voltage, so it is snapped and the snap error is logged.
- AND2/OR2 are simulated in SPICE as complete gates. Never add NAND2 + INV results.
- The same 200 corners are used for every gate and the test circuit, and that is what makes the
  Part C lookup in `GateLeakageModel` valid.

## Git
Branch per person/feature (`p1/runner`, `p3/gate-ml`, …). PR into `main` with one reviewer (pairs in
TODO.md). Don't commit `.venv/`, `.spice_work/` or pilot outputs. Do commit the final
`data/`, `artifacts/` and `results/` (they are deliverables).
