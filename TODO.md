# TODO: work split (4 people)

Team: Manya (P1), Joanna (P2), Sindhuja (P3), Gopal (P4), also in `configs/project.yaml → team.members`. Tick boxes as you go. Contracts are in
[SPEC.md](SPEC.md), conventions in [AGENTS.md](AGENTS.md).

| Person | Area | Owns | Deliverables |
|---|---|---|---|
| **P1** Manya | Transistor data quality + data checks (Part A.1) | `data/checks.py`, scripts 01–03, maintains `pdk.py`, `sampling.py`, `spice/`, `data/generate.py`, `plotting.py` | D1, D2 (FET), D3 |
| **P2** Joanna | ML transistor models (Part A.2) | `ml/*`, `metrics.py`, scripts 04–05 | D4, D5 |
| **P3** Sindhuja | Gate-level validation (Part B) | `gates/*`, scripts 06–09 | D2 (gates), D6, D7 |
| **P4** Gopal | Test circuit (Part C) + pipeline + report | `config.py`, `circuit/*`, scripts 00, 10–12, Makefile, `report/` | D2 (circuit), D8, D9, D10 |

---

## Day 1: everyone (15 min)

```bash
uv sync
make test                                   # 72 pass, 3 skipped (P4's circuit tests)
make bootstrap CONFIG=configs/dev.yaml QUICK=1
```
`bootstrap` takes about 20 s. It writes, under the `_dev_pilot` folders (gitignored, never deliverables):

| What | Path | Who uses it |
|---|---|---|
| pilot transistor datasets (500 rows/case) | `data/_dev_pilot/transistor/*.csv` | P2 |
| shared PVT corners (10 in quick mode) | `data/_dev_pilot/corners.csv` | P3, P4 |
| **stand-in** transistor models | `artifacts/_dev_pilot/models/*.joblib` | P3 |

P4 additionally runs `uv run python scripts/dev_placeholder_gate_ml.py --config configs/dev.yaml --quick`
to get a **placeholder** `gate_ml.csv` (made-up numbers, correct format).

`configs/dev.yaml` uses a **synthetic** model card (`tests/fixtures/fixture_45nm.pm`), so all of
this works before the TA assignment is filled in. Run everything with
`--config configs/dev.yaml [--quick]` until your real inputs exist. Then switch to
`configs/project.yaml`, with no code changes.

## How everyone works without waiting

| Person | Needs | Use now (stand-in) | Real one arrives when |
|---|---|---|---|
| P1 Manya | assigned model card | fixture card (`configs/dev.yaml`) | TAs give the file → `pdk/` |
| P2 Joanna | transistor datasets | `make data CONFIG=configs/dev.yaml [QUICK=1]` (500 or 2000 rows) | P1 runs step 02 on `project.yaml` |
| P3 Sindhuja | corners, transistor models | dev corners + stand-in models (bootstrap) | P2 ships `artifacts/models/*.joblib` |
| P4 Gopal | corners, gate subckts, gate-ML table | dev corners, `render_gate_subckt`, placeholder `gate_ml.csv` | P3 ships `data/gates/gate_ml.csv` |

The interfaces between the stages are fixed and tested (SPEC §4.6, §5.4, §6.3). Swapping a
stand-in for the real thing means **re-running a script, not changing code**.

---

## Foundation: DONE (do not re-implement; fix bugs in place)
- [x] `config.py`: `load_config` (validation, `quick` → pilot paths), `ensure_dirs`, all path helpers, `configs/dev.yaml`
- [x] `io.py`: `read_csv` / `write_csv` / `save_table` (bit-string-safe, full precision)
- [x] `pdk.py`: parse PTM card, nominal params, render per-sample `.model` blocks
- [x] `sampling.py`: `make_rng`, `draw_transistor_samples`, `draw_corners`, `corner_process_params`
- [x] `spice/`: `common` (header/options/model/control blocks), `runner` (parallel, failure detection), `parse`, `fet_netlist`
- [x] `data/generate.py`: simulate, top up failures deterministically, save/load with column contract
- [x] `metrics.py`: `relative_error`, `r2_log`, `summarize`, `results_table` (SPEC §7)
- [x] `plotting.py`: `save_fig`, `parity_axes`, `parity_plot_log`, `trend_scatter`
- [x] `ml/predictor.py`: model-file format (`save_bundle`), `TransistorModel`, `TransistorModelBank`, `select_case(s)`
- [x] `ml/baseline.py` + `scripts/dev_train_baseline.py`: STAND-IN models (crude by design, ~50 % error)
- [x] `gates/topology.py`: `GATE_LIBRARY` (all 5 gates), `logic_function`, `input_levels`, `terminals_on`, `validate_topology`
- [x] `gates/netlist.py`: `render_gate_subckt`, `subckt_pins`, `gate_widths`. All 18 gate cases are verified in ngspice (`test_gate_subckt_logic`)
- [x] `gates/ml_model.py`: `GateLeakageModel` (P4's lookup)
- [x] scripts 00 (env check), 01 (nominals), 02 (transistor data), 06 (corners), `dev_*` helpers; `make bootstrap`
- [x] Tests for all of the above (`make test`)

---

## Milestones
- [ ] **M0: Setup (day 1, everyone)**
  - [x] Fill `configs/project.yaml` with the TA assignment (model file, node, Lmin, VDD, test circuit, names)
  - [x] Put the assigned PTM file in `pdk/`, then `make check` (writes `artifacts/env.json`)
  - [ ] Everyone runs the Day 1 commands above
  - [ ] Agree on the git workflow (branch per person, PRs, reviewer pairs below)
- [ ] **M1: Every stage runs end-to-end on dev/stand-in inputs (end of week 1)**
- [ ] **M2: Part A final on the assigned PDK (week 2)**: D1, D3, D4, D5
- [ ] **M3: Part B final with P2's real models (week 3)**: D6, D7
- [ ] **M4: Part C final with P3's real gate ML (week 3–4)**: D8, and `make all` regenerates everything from scratch
- [ ] **M5: Report (week 4)**: D10 assembled, proofread, submitted

---

## P1 (Manya): transistor data quality + data checks

The generator already works. P1's job is to make the data **correct, sufficient and defended**.

### Data check (D3)
- [x] `data/checks.py`: `kirchhoff_residual`, `kirchhoff_table` → `results/tables/kirchhoff.{csv,md}`
- [x] the 3 trend plots (`plotting.trend_scatter` exists): Ioff vs T, Ioff vs vth0, Ig vs toxe
- [x] `scripts/03_data_check.py`
- [x] With the dev data: check the plots show the expected physics (Ioff grows ~exponentially with T and falls with vth0; Ig falls steeply with toxe)

### Real datasets (D1, D2-FET)
- [x] When the assigned card arrives: `make nominals data`, then inspect `artifacts/nominals.json` against the file by hand
- [x] Check that a re-run gives byte-identical CSVs (determinism)
- [x] ~~Generate the final data inside Docker~~ Dropped (team decision): final data is generated locally with ngspice-47
- [x] Review the failure handling on the real card (convergence failures? which corners?)

### Dataset size ("smallest dataset possible", brief Part A.1)
- [ ] Learning curve with P2 (n = 1k, 2k, 5k, 10k → validation p95 error). Pick n and justify it
- [ ] *(optional)* smarter sampling (e.g. oversample low |Vds| or high T where the error is largest)

### Speed (helps P4)
- [ ] *(if step 10 is slow)* batch several `op` runs per ngspice process (`alter`/`altermod` in `.control`), and check that the numbers are identical to separate runs

### Report sections
- [ ] *Data generation* (sampling, netlists, split summary from P2, data-check results), simulator + version for *Setup*

---

## P2 (Joanna): ML transistor models

`metrics.py` and the predictor API are done. Start on dev data now:
`make data CONFIG=configs/dev.yaml` (2000 rows/case, about 1 min) or the QUICK=1 pilot.

- [x] `ml/split.py`: split + save/load indices (written by P1)
- [x] `ml/features.py`: `build_features` (physics features, PMOS sign-flip), `TargetTransform` (log|I| + sign, or asinh). Justify the choice
- [x] `ml/zoo.py`: 4 model types, defaults, small search spaces
- [x] `ml/train.py`: `fit_one`, `compare_models`, `comparison_table`, `select_best`, `evaluate_on_test`, `save_model`
      **Save with `predictor.save_bundle`**, and put the regressor wrapper class in `leakml/ml/` (see `ml/baseline.py`) → `ml/regressor.py`
- [x] `scripts/04_train_compare.py` → D4 table (4 models × 4 cases, R², median, p95 on **val**)
- [x] `scripts/05_finalize_models.py` → D5 (test median/p95, pred-vs-SPICE plots with `plotting.parity_axes`, 4 model files)
- [x] Hand P3 a first real bank as early as possible: tuned models on 32nm_LP in `artifacts/models/`
- [ ] Improve accuracy: test mean accuracy is 73–96 % per current vs the ≥ 99 % target. Look at the worst-predicted samples (low Vds? high T? `is` vs `id`? the ~1e-14 A floor in OFF `id`)
- [ ] Learning curve together with P1 (P1 can call `train.fit_one` on nested subsets with the chosen model per case)
- [ ] Report: *Models*; the Part A part of *Error analysis*; *Limitations and improvements* (lead)

---

## P3 (Sindhuja): gate-level validation (Part B)

Topology, subckts, corners and the P4 lookup are done. `tests/test_spice.py::test_gate_subckt_logic`
is a working hand-built gate deck to start from.

### SPICE reference
- [ ] `gates/netlist.py`: `render_gate_netlist` (all helpers exist; see its docstring)
- [ ] `gates/spice_ref.py`: `extract_gate_result` (logic_ok check), `run_gate_reference`
- [ ] `scripts/07_gate_spice.py` → `gate_spice.csv` + example decks `gate_{GATE}.sp`
- [ ] **Freeze `data/corners.csv`** on the real config once step 07 has run (step 06 refuses to change it afterwards)

### ML prediction (develop against the stand-in bank)
- [ ] `gates/ml_model.py`: `resolve_node_voltages`, `fet_feature_rows` (vg snap + snap error), `leakage_from_currents` (SPEC §3.5, `topology.terminals_on`), `predict_gate_leakage`, `run_gate_ml` (one batched predict)
- [ ] `scripts/08_gate_ml.py`
- [ ] Sanity check that doesn't depend on model quality: feed SPICE-exact currents through `leakage_from_currents` and you must get `i_vdd_spice` back
- [ ] *(stretch)* `solve_internal_nodes_ml` (SPEC §5.5)

### Results (D6, D7)
- [ ] `gates/report.py`: `merge_spice_ml`, `gate_results_table` (`metrics.results_table`), `gate_parity_plot`, `worst_cases`
- [ ] `scripts/09_gate_report.py`
- [ ] Report: *Gate results*; the gate part of *Error analysis* (stacking effect, vg snap at n1, ...)

---

## P4 (Gopal): test circuit (Part C) + pipeline + report

Config, env check, Makefile are done. Develop with dev corners + the placeholder `gate_ml.csv`.

### Circuit
- [x] `circuit/definition.py`: `xor2`, `build_cla3` **or** `build_parity8` (the assigned one), `get_circuit`, `validate_circuit`
- [x] `circuit/logic.py`: `all_input_vectors`, `evaluate`, `gate_input_bits`, `reference_function`
- [x] `tests/test_circuit.py`: remove the skip, exhaustive logic vs reference
- [x] `circuit/netlist.py`: `render_circuit_netlist` (use `render_gate_subckt` + `subckt_pins` + `gate_widths`)
- [x] `circuit/spice_ref.py`: `run_circuit_reference` with checkpoint/resume (the longest SPICE job)
- [x] `scripts/10_circuit_spice.py` → `circuit_spice.csv` + `circuit_{name}.sp`
- [x] `circuit/ml_model.py`: `predict_circuit_leakage`, `run_circuit_ml` (`GateLeakageModel.from_csv`)
- [x] `scripts/11_circuit_ml.py`; `circuit/report.py` + `scripts/12_circuit_report.py` → D8
- [x] *(nice-to-have)* `loading_error_analysis`

### Integration + report (D9, D10)
- [ ] Final clean run: `make all` from an empty `data/ results/ artifacts/`, and record the wall time
- [ ] `report/report.md`: assemble everyone's sections, *Setup*, *Test-circuit results*, final proofread
- [ ] Check the submission against brief §6 (the deliverables table in SPEC §10)

---

## Review pairs (every PR needs one approval)
| Author | Reviewer | Why |
|---|---|---|
| P1 Manya | P3 Sindhuja | P3 is the next heaviest SPICE user |
| P2 Joanna | P1 Manya | P1 knows the data and its sign conventions |
| P3 Sindhuja | P2 Joanna | the ML gate code consumes P2's API |
| P4 Gopal | P3 Sindhuja | the circuit reuses P3's subckts/topology |

## Shared / anyone
- [ ] Confirm with the TAs: leakage definition (SPEC §3.5), fixed gate sizing (§5.2), the accuracy definition (§7), ngspice as the simulator
- [ ] Keep `TODO.md` boxes up to date (tick them in your PRs)
