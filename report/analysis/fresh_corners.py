"""Fresh-corner generalisation check (scratchpad only; never writes to the repo's data/ or results/).

Draws a NEW set of n_corners PVT corners with seed cfg.seeds.corners + 1000 through the same
sampling.draw_corners call as step 06, runs the step-07 SPICE reference (run_gate_reference) and the
step-08 ML (run_gate_ml) on them, and builds the step-09 tables (merge_spice_ml, gate_results_table,
worst_cases). Run from the worktree root:  uv run python fresh_corners.py <out_dir>
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from leakml import io
from leakml.config import load_config
from leakml.gates.ml_model import run_gate_ml
from leakml.gates.report import gate_results_table, merge_spice_ml, worst_cases
from leakml.gates.spice_ref import run_gate_reference
from leakml.ml.predictor import TransistorModelBank
from leakml.pdk import load_model_card
from leakml.sampling import draw_corners, make_rng

os.environ.setdefault("OMP_WAIT_POLICY", "passive")  # as step 07
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config("configs/project.yaml")
card = load_model_card(cfg.assignment.model_file)
seed = cfg.seeds.corners + 1000

corners = draw_corners(cfg.gates.n_corners, cfg, card, make_rng(seed, "corners"))
orig = io.read_csv(cfg.paths.corners_csv)
assert not corners[["vdd", "temp"]].round(12).merge(orig[["vdd", "temp"]].round(12)).shape[0]
io.write_csv(corners, OUT / "corners_fresh.csv")
print(f"seed {seed}: {len(corners)} fresh corners")
print(corners.describe().T[["min", "max"]].to_string(float_format=lambda x: f"{x:.4g}"))

spice_path = OUT / "gate_spice_fresh.csv"
if spice_path.exists():
    gs = io.read_csv(spice_path)
else:
    t0 = time.time()
    gs = run_gate_reference(cfg, corners, card)
    io.write_csv(gs, spice_path)
    print(f"step 07 on fresh corners: {len(gs)} rows in {time.time() - t0:.0f} s")
print(
    f"failed sims {int(gs['i_vdd_spice'].isna().sum())}, logic_ok {int(gs['logic_ok'].sum())}/{len(gs)}"
)

bank = TransistorModelBank.load_all(cfg.paths.models_dir)
ml = run_gate_ml(cfg, gs, corners, bank)
io.write_csv(ml, OUT / "gate_ml_fresh.csv")
print(f"i_vdd_ml <= 0: {int((ml['i_vdd_ml'] <= 0).sum())}, direct <= 0: "
      f"{int((ml['i_vdd_ml_direct'] <= 0).sum())}")  # fmt: skip
print("snap", ml.groupby("gate", sort=False)["max_vg_snap_err"].max().to_dict())

merged = merge_spice_ml(gs, ml, n_corners=cfg.gates.n_corners)
io.write_csv(merged, OUT / "gate_results_fresh.csv")
io.save_table(gate_results_table(merged, cfg.ml.i_floor_a), OUT / "gate_results_table_fresh")
io.save_table(worst_cases(merged, 20, cfg.ml.i_floor_a), OUT / "gate_worst_cases_fresh")
print("done ->", OUT)
