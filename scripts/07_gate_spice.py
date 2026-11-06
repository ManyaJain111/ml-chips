"""Step 07: SPICE reference for 18 gate cases x 200 corners (Part B step 3).  Owner: P3.

Inputs:  data/corners.csv (step 06, shared with Part C), the assigned model card
Outputs: data/gates/gate_spice.csv
         netlists/examples/gate_{INV,NAND2,NOR2,AND2,OR2}.sp   (Deliverable 2: corner 0, all inputs 0)

Exits non-zero if any deck fails to simulate or any row has the wrong logic level. In that case
gate_spice.csv is NOT written; the full table (failures included) goes to
<simulator.work_dir>/gate_spice_FAILED.csv for debugging instead.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from leakml import io
from leakml.cli import setup_logging
from leakml.config import DEFAULT_CONFIG, ensure_dirs, load_config
from leakml.constants import GATE_INPUT_COMBOS, GATES
from leakml.gates.netlist import render_gate_netlist
from leakml.gates.spice_ref import run_gate_reference
from leakml.gates.topology import get_gate, logic_function
from leakml.pdk import load_model_card

log = logging.getLogger("07_gate_spice")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--quick", action="store_true", help="pilot run with small sample counts")
    return p.parse_args(argv)


def main(argv=None) -> None:
    # parallel ngspice processes starve each other's OpenMP threads otherwise (30-40x slower)
    os.environ.setdefault("OMP_WAIT_POLICY", "passive")
    args = parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, quick=args.quick)
    ensure_dirs(cfg)

    if not cfg.paths.corners_csv.exists():
        sys.exit(f"{cfg.paths.corners_csv} not found: run step 06 (make corners) first")
    corners = io.read_csv(cfg.paths.corners_csv)  # the SAME file as Part C
    card = load_model_card(cfg.assignment.model_file)
    print(f"{len(corners)} corners from {cfg.paths.corners_csv}, card {cfg.assignment.model_file}")

    # Deliverable 2: the corner-0 deck of each gate (all inputs at logic 0)
    c0 = corners[corners["corner_id"] == 0]
    if len(c0) != 1:
        sys.exit(f"{cfg.paths.corners_csv} must contain exactly one corner_id 0")
    c0_row = c0.iloc[0].to_dict()
    for g in GATES:
        deck = render_gate_netlist(get_gate(g), GATE_INPUT_COMBOS[g][0], c0_row, card, cfg)
        (cfg.paths.netlist_examples / f"gate_{g}.sp").write_text(deck)
    print(f"example decks -> {cfg.paths.netlist_examples}/gate_<GATE>.sp")

    df = run_gate_reference(cfg, corners, card)

    failed = df["i_vdd_spice"].isna()
    wrong = ~df["logic_ok"] & ~failed
    for r in df[wrong].itertuples(index=False):
        expected = r.vdd * logic_function(r.gate, r.inputs)
        log.error(
            "logic failure: %s %s corner %d: v_out=%.6g V, expected %.6g V (|err| >= 0.1*vdd=%.4g)",
            r.gate,
            r.inputs,
            r.corner_id,
            r.v_out_spice,
            expected,
            0.1 * r.vdd,
        )

    print(
        f"\nrows: {len(df)}   failed simulations: {int(failed.sum())}   "
        f"logic_ok: {int(df['logic_ok'].sum())}/{len(df)}"
    )
    summary = (
        df.groupby("gate", sort=False)["i_vdd_spice"].agg(["min", "median", "max"]).reindex(GATES)
    )
    print("i_vdd_spice (A) per gate:")
    print(summary.to_string(float_format=lambda x: f"{x:.3e}"))

    if failed.any() or wrong.any():
        dump = Path(cfg.simulator.work_dir) / "gate_spice_FAILED.csv"
        dump.parent.mkdir(parents=True, exist_ok=True)
        io.write_csv(df, dump)
        print(
            f"\nFAILED: {int(failed.sum())} simulation failures, {int(wrong.sum())} logic failures."
            f" {cfg.paths.gate_spice_csv} NOT written; full table (all rows) -> {dump}",
            file=sys.stderr,
        )
        sys.exit(1)

    io.write_csv(df, cfg.paths.gate_spice_csv)
    print(f"\nwrote {len(df)} rows -> {cfg.paths.gate_spice_csv}")


if __name__ == "__main__":
    main()
