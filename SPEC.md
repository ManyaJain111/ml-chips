# SPEC: ML-Driven PVT-Variation-Aware Leakage Estimation

Technical specification and **inter-module contracts** for the DVD course project
(brief: [DVD_ML_PROJECT.pdf](DVD_ML_PROJECT.pdf)). Code comments cite sections of this file as
`SPEC §x.y`. If code and spec disagree, the spec wins. Change the spec first, then the code, and
tell the affected owner (see [TODO.md](TODO.md)).

---

## 1. Goal

Estimate CMOS leakage under process/voltage/temperature (PVT) variation with ML instead of full
SPICE:

1. **Part A:** simulate single NMOS/PMOS transistors (ON and OFF) at random PVT, then train four ML
   models (one per case) that predict the terminal currents `Id, Ig, Is, Ib`.
2. **Part B:** predict the leakage of INV, NAND2, NOR2, AND2 and OR2 for every input combination
   using **only** those four models, and compare with transistor-level SPICE (18 cases × 200 corners).
3. **Part C:** build the assigned test circuit (3-bit CLA or 8-bit parity checker)
   hierarchically from the gate models, and compare with SPICE of the full netlist.

Target: **≥ 99 % mean accuracy** (definition in §7).

## 2. Assignment (`configs/project.yaml`)

| Key | Meaning |
|---|---|
| `assignment.model_file` | PTM BSIM4 card in `pdk/`, one of 16nm/22nm/32nm/45nm × HP/LP/MGK |
| `assignment.lmin` | feature size in metres. **L = Lmin for every FET** |
| `assignment.vdd_nominal` | 0.9 or 1.0 V |
| `assignment.test_circuit` | `cla3` or `parity8` |

`leakml.config.load_config()` refuses to run while any value is `TODO`.

`configs/dev.yaml` is the same config with a **synthetic** model card
(`tests/fixtures/fixture_45nm.pm`, 45 nm-like, VDD = 1.0 V) and `_dev` output folders. It lets
every stage be developed and tested before, and independently of, the real assignment (§11).
Nothing produced with it is a deliverable.

## 3. Conventions

### 3.1 Units
SI everywhere: volts, amperes, **metres** (W and L), temperature in **°C** (as SPICE `.temp`
expects). Report figures may rescale (nA, nm), but stored data never does.

### 3.2 Current sign convention
All stored currents are **positive when flowing INTO the device terminal**.
ngspice reports `i(Vx)` as current flowing from the `+` node through the source to the `-` node.
With `Vd d 0 dc …`, the current into the drain is therefore `id = -i(vd)`. KCL then gives
`id + ig + is + ib ≈ 0`, which is the Kirchhoff data check (Deliverable 3).

### 3.3 Node names
| Node | Name |
|---|---|
| supply | `vdd` (source `Vvdd`) |
| ground | `0` (pin `gnd` inside subckts) |
| gate inputs | `a`, `b` (input bit string `"AB"`: first char ; `a`) |
| gate output | `out` |
| NAND2 series-NMOS node | `x` |
| NOR2 series-PMOS node | `y` |
| AND2/OR2 first-stage output | `n1` |

### 3.4 Randomness
Every random draw comes from `sampling.make_rng(seed, stream_label)` with seeds from
`cfg.seeds`. Each Part A case gets its own stream (`make_rng(seeds.transistor_data, case)`), so
regenerating one case does not change the others. No unseeded `np.random.*` / `random.*` anywhere.

### 3.5 Leakage definition
**Primary metric, `i_vdd`:** the static current delivered by the VDD supply.
* SPICE: `i_vdd_spice = -i(vvdd)`
* ML: `i_vdd_ml = Σ (current into FET terminal)` over every FET terminal connected to net `vdd`
  (in our topologies, the PMOS sources and PMOS bulks tied to `vdd`).

**Secondary metric, `i_in`:** the current delivered by the input sources at logic 1 (gate tunnelling
into the driven gates). SPICE: `-Σ i(v<in>)` over the high inputs. ML: the sum of `ig` of FETs gated
by those inputs. Recorded so the definition can be switched if the TAs want total leakage
`i_vdd + i_in` (open question Q1, §11).

## 4. Part A: transistor models

### 4.1 Sampling (`leakml.sampling`)
Per case: `n_samples_per_case` (30,000; was 10,000, raised after the learning curve showed the
error still falling at 7k training rows) rows, every parameter independent, **VDD first**:

| Parameter | Distribution |
|---|---|
| `vdd` | U(0.9·VDDnom, 1.1·VDDnom) |
| `vd`, `vs` | U(0, vdd), using that row's vdd (ON cases: see low-\|Vds\| note below) |
| `w` | U(1, 8) · Lmin |
| `temp` | U(−40, 125) °C |
| `vth0, toxe, ndep, u0, xj` | nominal·(1 + U(−0.1, 0.1)), nominals parsed from the device's `.model` block. PMOS `vth0` is negative and keeps its sign |
| `vg`, `vb` | fixed by the case (table below), `vdd` = that row's vdd |

**Low-|Vds| oversampling (ON cases only).** Inside a gate every ON FET sits at |vds| ≈ 1e-7 V
(drain and source pulled to the same rail), which uniform vd, vs almost never produce (~0.2 % of
rows with |vds| < 1 mV). In `nmos_on` and `pmos_on`, a random 40 % of rows
(`sampling.LOW_VDS_FRAC`) instead get `vd = vs ± |vds|` with |vds| log-uniform in [1e-9, 1e-2] V
(`LOW_VDS_RANGE`) and a random sign, reflected to stay in [0, vdd]. vd and vs of these rows are
rounded to the deck's 10 significant digits so the stored inputs equal the simulated ones. These
draws come after all others, so the uniform rows and the OFF cases are unchanged.

**Source pinning (OFF cases only).** Inside a gate an OFF FET sits with its source on its own
rail (NMOS: 0, PMOS: vdd, i.e. at its body) and the drain up to the other rail; stacked OFF FETs
see a small source offset. Uniform vd, vs put fewer than 1 % of rows near that point. In
`nmos_off` and `pmos_off`, a random 40 % of rows (`sampling.SOURCE_PIN_FRAC`) instead get
`vs` = rail ± offset, offset log-uniform in [1e-9, 1e-1] V (`SOURCE_PIN_RANGE`), towards the inside
of [0, vdd], rounded to the deck's 10 significant digits; `vd` stays uniform. Drawn after all
others, so the other rows and the ON cases are unchanged.

**Boundary oversampling (all cases).** Tree models are least accurate near the edges of the sampled
box (on the 10k data, OFF-case validation p95 ≈ 58 % for rows within 5 % of an edge vs ≈ 45 %
inside), and the corners reach those edges too. A random 30 % of rows (`sampling.BOUNDARY_FRAC`)
are boundary rows: on each, every one of `vdd, temp, w, vth0, toxe, ndep, u0, xj`
(`BOUNDARY_INPUTS`) is, with probability ½ (`BOUNDARY_PER_INPUT`), redrawn uniformly from the
outer 5 % of its range (`BOUNDARY_BAND`) at a random end. So ~24 % of rows lie within 5 % of each
input's edge instead of 10 %. Nothing leaves the ranges above. vdd is set before vd, vs are drawn,
so vd, vs ≤ vdd still holds. These draws use a child stream of the case RNG, so the low-|Vds| and
source-pinning overrides are unaffected. Validation and test are drawn the same way, so their
metrics include the boundary rows.

| Case | vg | vb |
|---|---|---|
| `nmos_on` | VDD | 0 |
| `nmos_off` | 0 | 0 |
| `pmos_on` | 0 | VDD |
| `pmos_off` | VDD | VDD |

### 4.2 Netlists (all decks)
Every deck (FET, gate, circuit) contains, verbatim:
```
.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9
```
and runs one DC operating point (`op`) with `.temp` set to the sample/corner temperature.
Process parameters are applied by **rewriting the `.model` block text** with the drawn values
(`pdk.render_model_card`). Single-FET decks give **every terminal its own voltage source**
(`Vd`, `Vg`, `Vs`, `Vb`). Print precision: `set numdgt=12`.

Example single-FET deck structure:
```
* nmos_off sample 17
.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9
.temp 87.3
.model nmos nmos level=54 ... vth0=0.4712 toxe=1.08e-9 ...
Vd d 0 dc 0.613
Vg g 0 dc 0
Vs s 0 dc 0.120
Vb b 0 dc 0
M1 d g s b nmos W=8.8e-08 L=2.2e-08
.control
set numdgt=12
op
print i(vd) i(vg) i(vs) i(vb)
.endc
.end
```

### 4.3 Dataset schema (Deliverable 1)
`data/transistor/{nmos_on,nmos_off,pmos_on,pmos_off}.csv`, `n_samples_per_case` rows each, columns **in this
order** (`constants.DATASET_COLUMNS`):

`vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj, id, ig, is, ib`

Failed simulations are dropped and topped up from the same RNG stream, so each file has exactly
n rows. The failure count is logged.

### 4.4 Data check (Deliverable 3)
* Kirchhoff: residual `r = id+ig+is+ib` per row. Report median/p95/max of `|r|` and of
  `|r| / max(|id|,|ig|,|is|,|ib|)` per case ; `results/tables/kirchhoff.*`.
* Trend plots: OFF current vs temperature, OFF current vs vth0, gate current vs toxe ; `results/plots/trend_*`.

### 4.5 Models (Deliverables 4, 5)
* Split: shuffle with `seeds.split`, 70/15/15 (or 80/10/10). The indices are saved to `artifacts/splits/`.
* **Four model types**, compared on **validation**, defaults in `ml/zoo.py`: polynomial ridge,
  random forest, gradient-boosted trees (LightGBM), MLP.
* Targets: currents span ~10 decades and change sign, so regress in a transformed space
  (`log10|I|` + sign, or `asinh(I/I0)`). Features may include physics-derived terms (`vds`,
  `vgs`, `1/T`, `vgs/vT`, normalised process params). The fitted feature list and transform are
  saved inside the model file.
* Selection: lowest `cfg.ml.selection_metric` (default p95 relative error) on validation.
  **The test split is used only once**, in `scripts/05_finalize_models.py`.
* Saved model: `artifacts/models/{case}.joblib` in the bundle format of §4.6.

### 4.6 Predictor API (contract P2 ; P3/P4)
```python
bank = TransistorModelBank.load_all(Path("artifacts/models"))
case = TransistorModelBank.select_case(device="nmos", vg=0.0, vdd=0.9)  # -> Case.NMOS_OFF
I = bank.predict(X, cases)  # X: DataFrame[INPUT_COLUMNS], raw values
# -> DataFrame[id, ig, is, ib] in A, into-device sign, same index
```
`X.vg` must be exactly 0 or `vdd` (as in training). The caller snaps it
(`predictor.snapped_gate_voltage`) and reports the snap error.

**Model file ("bundle") format, `leakml.transistor_model/v1`**: a joblib dump written by
`predictor.save_bundle(path, case, regressor, meta)`:
```python
{"format": "leakml.transistor_model/v1", "case": "nmos_off",
 "regressor": obj,   # obj.predict(X: DataFrame[INPUT_COLUMNS]) -> ndarray (n, 4), amperes,
                     # TARGET_COLUMNS order, into-device sign. Owns ALL preprocessing.
 "meta": {"model_name": ..., "train_ranges": {col: [lo, hi]}, ...}}
```
The regressor's class must live inside the `leakml` package (so it unpickles from any script).
With `meta["train_ranges"]` present, `predict` warns on extrapolation. `ml/baseline.py` is a
minimal working example, and it is a STAND-IN, not a deliverable.

## 5. Part B: gate-level validation

### 5.1 Corners (`data/corners.csv`)
200 corners, drawn once with `seeds.corners` and **shared by Part B and Part C**. One corner is
one draw of `vdd`, `temp`, one NMOS process draw (`n_vth0 … n_xj`) and one PMOS process draw
(`p_vth0 … p_xj`). All NMOS in a gate/circuit share the NMOS draw, and all PMOS share the PMOS draw.
Columns: `constants.CORNER_COLUMNS`. **Do not regenerate after Part B/C runs exist.**

### 5.2 Gate topologies (`gates/topology.py`)
Standard static CMOS. Fields are (d, g, s, b). PMOS bulk = `vdd`, NMOS bulk = `0`.
Sizing is fixed for all gates and circuits: NMOS W = `nmos_w_mult`·Lmin, PMOS W = `pmos_w_mult`·Lmin
(defaults 2 and 4, which must lie inside the training range 1–8).

| Gate | FETs (d, g, s, b) | Internal nodes |
|---|---|---|
| INV | MP(out,a,vdd,vdd) · MN(out,a,0,0) | none |
| NAND2 | MPA(out,a,vdd,vdd) · MPB(out,b,vdd,vdd) · MNA(out,a,x,0) · MNB(x,b,0,0) | x |
| NOR2 | MPA(y,a,vdd,vdd) · MPB(out,b,y,vdd) · MNA(out,a,0,0) · MNB(out,b,0,0) | y |
| AND2 | NAND2 with out;n1, then INV: MP2(out,n1,vdd,vdd) · MN2(out,n1,0,0) | x, n1 |
| OR2 | NOR2 with out;n1, then INV: MP2(out,n1,vdd,vdd) · MN2(out,n1,0,0) | y, n1 |

18 cases: INV {0,1}, the others {00,01,10,11}. Inputs are driven at **exactly** 0 or VDD.
AND2/OR2 are simulated as **complete** gates in SPICE (never NAND2+INV added together).

### 5.3 ML gate prediction (`gates/ml_model.py`)
For each (gate, inputs, corner):
1. Node voltages: `vdd` ; corner VDD, `0` ; 0, inputs ; 0/VDD, `out`/`x`/`y`/`n1` ; **the
   SPICE-recorded values** for that same (gate, inputs, corner). SPICE currents are never read.
2. Per FET: (vd, vg, vs, vb) from its nodes, case from (device, gate logic level), `vg` snapped to
   0/VDD, W from the sizing, and temp/vdd/process params from the corner (n_* or p_*).
3. One batched `bank.predict` call for all FETs of all 3,600 cases.
4. `i_vdd_ml` and `i_in_ml` per §3.5, after the KCL correction below (`i_vdd_ml_direct` is the
   same sum without it).

**KCL correction (`kcl_corrected_currents`, `KCL_VDS_MAX = 1e-3` V).** An ON FET inside a static
gate sits between two nodes at almost the same voltage (|vds| ~ 1e-9..1e-4 V on 32nm_LP) and carries
only leakage-sized current set by the rest of the gate. The Part A models were trained on
vd, vs ~ U(0, VDD) and cannot resolve that current: their absolute error at vds ~ 0 is 10-1000x the
true value. So, for every ON FET (`nmos_on`/`pmos_on`) with |vd - vs| < `KCL_VDS_MAX`:
- the model's `ig` and `ib` are kept; `id` is the unknown and `is = -(id + ig + ib)`;
- the unknown `id`s are solved from KCL at the floating nodes (`out`, `x`, `y`, `n1`): the currents
  into all FET terminals on each node sum to zero. Every other FET keeps its model currents.
- Rows with the same pattern of corrected FETs are solved together with the pseudo-inverse
  (least-squares, minimum-norm). For ON FETs in parallel (e.g. both PMOS of NAND2 `00`) only their
  sum is determined, and the minimum-norm solution splits it equally. Any other split differs by a
  current circulating in the parallel loop, which does not change `i_vdd`. This is checked for every
  pattern (`ValueError` otherwise, which can only happen with an all-ON vdd-to-ground path).
1 mV is far above any in-gate ON |vds| and far below a genuinely biased ON device (NOR2 `10`: MPB at
|vds| ~ 0.2 V keeps its ML current). `i_in_ml` is unchanged, because `ig` is never corrected.
`i_vdd_ml` (KCL) is the value used for Part B results and by `GateLeakageModel` in Part C.

### 5.4 Gate result schema
`data/gates/gate_spice.csv`: `gate, inputs, corner_id, vdd, temp, i_vdd_spice, i_in_spice,
v_out_spice, v_x_spice, v_y_spice, v_n1_spice, logic_ok`. Missing nodes are NaN.
`data/gates/gate_ml.csv`: `gate, inputs, corner_id, i_vdd_ml, i_in_ml, v_x_ml, v_y_ml, v_n1_ml, max_vg_snap_err`.
It also has an extra last column `i_vdd_ml_direct`, the §3.5 sum without the KCL correction (§5.3), kept for comparison.
`data/gates/gate_results.csv`: 1:1 join on `(gate, inputs, corner_id)`, exactly 3,600 rows.

### 5.5 Stretch: ML-solved internal nodes
With `gates.node_voltage_source: ml_solve`, internal node voltages are found by solving KCL with the
ML currents (brentq/root on [0, VDD]) instead of reading SPICE. This is optional. The brief only
requires the SPICE-voltage mode, but it would give a fully SPICE-free gate model.

## 6. Part C: test circuit

### 6.1 Circuit definitions (`circuit/definition.py`)
Gate-level netlists use **only** INV/NAND2/NOR2/AND2/OR2. XOR2 is built from 4×NAND2.
* `cla3`: inputs `a0..a2, b0..b2, cin` (7 bits ; 128 vectors). `g_i = a_i·b_i`, `p_i = a_i ⊕ b_i`,
  carries c1..c3 computed directly from g/p/cin (lookahead, not ripple) with 2-input gates,
  `s_i = p_i ⊕ c_i`. Outputs `s0..s2, cout`.
* `parity8`: inputs `d0..d7` (8 bits ; 256 vectors). A balanced XOR tree of 7 XOR2 (28 NAND2).
  Output = XOR of all bits.

The same definition drives both the SPICE expansion and the ML composition. The logic simulation
is unit-tested exhaustively against a behavioural reference (integer add / popcount parity).

### 6.2 SPICE reference
One flat transistor-level deck per (vector, corner), built from the Part B gate subckts with the
same sizing and the same corners file. 128×200 = 25,600 (CLA) or 256×200 = 51,200 (parity) decks.
Inter-gate loading is physically present here.

### 6.3 Hierarchical ML
For each vector: logic-simulate ; (gate type, input bits) per instance ; `GateLeakageModel.predict(gate,
bits, corner_id)` (the Part B ML result for that gate/inputs/corner) ; sum. Inputs are assumed
exactly 0/VDD and loading is ignored, so expect a lower accuracy than at gate level.

### 6.4 Circuit result schema
`circuit_spice.csv`: `circuit, vector, corner_id, vdd, temp, i_vdd_spice, i_in_spice, logic_ok`.
`circuit_ml.csv`: `circuit, vector, corner_id, i_vdd_ml, i_in_ml, n_<gate>_<bits>…`.
`circuit_results.csv`: 1:1 join on `(circuit, vector, corner_id)`.

## 7. Metrics (`leakml/metrics.py`)
With `e = (ML − SPICE) / |SPICE|` per sample:

| Name | Definition |
|---|---|
| SPICE median / ML median | median of `i_spice` / `i_ml` |
| bias | `100 · median(e)` % (signed) |
| median error | `100 · median(|e|)` % |
| p95 error | `100 · P95(|e|)` % |
| max error | `100 · max(|e|)` % (for error analysis) |
| mean accuracy | `100 · (1 − mean(|e|))` %. **Target ≥ 99 %** |
| R² (Part A) | on `log10|I|` |

Samples with `|SPICE| < cfg.ml.i_floor_a` (default 1e-15 A) are excluded from relative errors and
counted (`n_excluded`). This only matters for terminal currents that pass through zero (e.g. `id`
when `vd ≈ vs`), not for gate/circuit leakage.

## 8. File layout and artifacts
```
configs/project.yaml          assignment + all knobs
pdk/<model>.pm                assigned PTM card (not shipped, see pdk/README.md)
src/leakml/                   library (see package docstrings)
scripts/NN_*.py               pipeline steps, run in order (Makefile)
data/transistor/{case}.csv    D1
data/corners.csv              shared corners
data/gates/gate_{spice,ml,results}.csv
data/circuit/circuit_{spice,ml,results}.csv
netlists/examples/            D2: fet_{case}.sp, gate_{GATE}.sp, circuit_{name}.sp
artifacts/env.json            simulator + versions (report Setup)
artifacts/nominals.json       parsed nominal process params
artifacts/splits/{case}.npz
artifacts/models/{case}.joblib   D5 saved models
results/tables/*.{csv,md}     D3 kirchhoff, D4 model_comparison, D5 chosen_models_test, D6 gate_results, D8 circuit_results
results/plots/*.{png,pdf}     D3 trends, D5 pred_vs_spice_{case}, D7 gate_parity, D8 circuit_parity
report/                       D10
```
`--quick` runs write to `data/_pilot/`, `artifacts/_pilot/` and `results/_pilot/` (example netlists
in `results/_pilot/netlists/`) and never overwrite deliverables. `configs/dev.yaml` writes to
`*/_dev/` (and `*/_dev_pilot/` with `--quick`). All of these are gitignored.

Every path comes from `cfg.paths` (e.g. `cfg.paths.gate_ml_csv`, `cfg.paths.model_file(case)`).
Read and write CSVs with `leakml.io.read_csv` / `write_csv`. Plain `pd.read_csv` turns the bit
strings `"01"` / `"0010110"` into integers.

## 9. Pipeline and runtime budget (rough, 8 workers)
| Step | Script | Owner | Sims | Est. time |
|---|---|---|---|---|
| env check | 00 | P4 | 1 | s |
| nominals | 01 | P1 | – | s |
| transistor data | 02 | P1 | 40,000 | ~5 min |
| data check | 03 | P1 | – | s |
| train + compare | 04 | P2 | – | min–tens of min |
| finalize | 05 | P2 | – | s |
| corners | 06 | P3 | – | s |
| gate SPICE | 07 | P3 | 3,600 | ~1 min |
| gate ML | 08 | P3 | – | s |
| gate report | 09 | P3 | – | s |
| circuit SPICE | 10 | P4 | 25,600 / 51,200 | ~10–30 min |
| circuit ML | 11 | P4 | – | s |
| circuit report | 12 | P4 | – | s |

ngspice process start-up dominates. If step 10 is too slow, batch several `op` runs per ngspice
process (`.control` loop with `alter`/`altermod`), but only after checking the numbers are identical.

## 10. Deliverables ; where they come from
| # | Deliverable | Produced by |
|---|---|---|
| 1 | 4 CSV datasets | 02 ; `data/transistor/` |
| 2 | example netlists | 02, 07, 10 ; `netlists/examples/` |
| 3 | Kirchhoff check + 3 trend plots | 03 |
| 4 | model comparison table (4×4, R², median, p95 on val) | 04 |
| 5 | chosen models: test median/p95, pred-vs-SPICE plots, 4 model files | 05 |
| 6 | gate results table (18 rows) | 09 |
| 7 | gate parity plot (log) | 09 |
| 8 | test-circuit table + parity plot | 12 |
| 9 | code: ordered scripts, fixed seeds | `make all` |
| 10 | report | `report/` |

## 11. Working independently: stand-ins
Each stage can be built and tested before its upstream stage is finished. Stand-ins have the
**exact** format of the real artifact, so switching to the real one means re-running a script,
never changing code.

| Real artifact (producer) | Stand-in | Made by |
|---|---|---|
| assigned PTM card (TAs) | `tests/fixtures/fixture_45nm.pm` via `configs/dev.yaml` | committed |
| transistor datasets (P1, step 02) | the same generator on the dev config | `make data CONFIG=configs/dev.yaml [QUICK=1]` |
| corners (P3, step 06) | dev corners | `make corners CONFIG=configs/dev.yaml [QUICK=1]` |
| transistor models (P2, step 05) | `ml/baseline.py` ExtraTrees (crude, ~50 % error) | `scripts/dev_train_baseline.py` |
| gate-ML table (P3, step 08) | smooth made-up function, flagged `source=placeholder` | `scripts/dev_placeholder_gate_ml.py` |

`make bootstrap CONFIG=configs/dev.yaml QUICK=1` builds the first four stand-ins in about 20 s.
The dev scripts refuse to write into the real `data/` or `artifacts/` folders.
