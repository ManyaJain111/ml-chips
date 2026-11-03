"""leakml.spice + FET decks + gate subckts (implemented foundation). Tests marked requires_ngspice run
the real simulator; they skip automatically where ngspice is not installed."""

import re
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from leakml.constants import CASES, GATE_INPUT_COMBOS, GATES, SPICE_OPTIONS
from leakml.data.generate import generate_case_dataset
from leakml.gates.netlist import gate_widths, render_gate_subckt, subckt_pins
from leakml.gates.topology import get_gate, input_levels, logic_function
from leakml.sampling import draw_transistor_samples, make_rng
from leakml.spice.common import END, model_block, netlist_header, op_control_block
from leakml.spice.fet_netlist import FET_EXPECT, FET_SOURCES, render_fet_netlist
from leakml.spice.parse import currents_into_device, parse_print_output
from leakml.spice.runner import run_many_cfg, run_netlist

requires_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")

# Real ngspice-47 batch output (op + print), captured while building the scaffold.
CAPTURED = """
Note: No compatibility mode selected!

Circuit: * nmos off probe

Doing analysis at TEMP = 87.300000 and TNOM = 27.000000

Using SPARSE 1.3 as Direct Linear Solver

No. of Data Rows : 1
i(vd) = -6.75118037774e-09
i(vg) = 6.762182953400e-09
i(vs) = -1.10470977521e-11
i(vb) = 4.452209531968e-14
v(x1.x) = 6.719982934379e-01
Note: Simulation executed from .control section
"""


def test_parse_print_output():
    v = parse_print_output(CAPTURED)
    assert v == {
        "i(vd)": -6.75118037774e-09,
        "i(vg)": 6.762182953400e-09,
        "i(vs)": -1.10470977521e-11,
        "i(vb)": 4.452209531968e-14,
        "v(x1.x)": 6.719982934379e-01,
    }


def test_currents_into_device_sign():
    cur = currents_into_device(parse_print_output(CAPTURED), FET_SOURCES)
    assert cur["id"] == pytest.approx(6.75118037774e-09)  # into the drain = -i(vd)
    with pytest.raises(KeyError):
        currents_into_device({}, FET_SOURCES)


@pytest.fixture
def one_sample(dev_cfg, card):
    return draw_transistor_samples("nmos_off", 1, dev_cfg, card, make_rng(0, "t")).iloc[0].to_dict()


def test_fet_deck_structure(one_sample, card, dev_cfg):
    deck = render_fet_netlist(one_sample, "nmos_off", card, dev_cfg.assignment.lmin)
    assert SPICE_OPTIONS in deck
    assert deck.startswith("*")
    sources = re.findall(r"^V(\w)\s+(\w+)\s+0\s+dc", deck, flags=re.M)
    assert sorted(sources) == [("b", "b"), ("d", "d"), ("g", "g"), ("s", "s")]  # one per terminal
    assert re.search(r"^M1 d g s b nmos W=\S+ L=4\.5e-08$", deck, flags=re.M)
    assert deck.count(".model") == 1  # only the needed device


def test_print_chunking():
    blk = op_control_block([f"v(n{i})" for i in range(20)])
    assert blk.count("print ") == 3 and "set numdgt=12" in blk


@requires_ngspice
@pytest.mark.parametrize("case", CASES)
def test_kcl_on_real_runs(case, dev_cfg, card):
    s = draw_transistor_samples(case, 8, dev_cfg, card, make_rng(5, case.value))
    decks = [
        render_fet_netlist(r, case, card, dev_cfg.assignment.lmin) for r in s.to_dict("records")
    ]
    for res in run_many_cfg(decks, dev_cfg, expect=FET_EXPECT, progress=False):
        assert res.ok, res.error
        cur = currents_into_device(res.values, FET_SOURCES)
        assert abs(sum(cur.values())) / max(map(abs, cur.values())) < 1e-6


@requires_ngspice
def test_failure_is_detected_and_deck_kept(dev_cfg):
    res = run_netlist(
        "* bad\nM1 d g s b nosuchmodel W=1u L=1u\n.control\nop\nprint i(vd)\n.endc\n.end\n",
        work_dir=dev_cfg.simulator.work_dir,
        expect=["i(vd)"],
    )
    assert not res.ok and res.error
    assert res.netlist_path is not None and res.netlist_path.exists()
    res.netlist_path.unlink()


@requires_ngspice
def test_run_many_preserves_order(dev_cfg):
    decks = [
        "\n".join(
            [
                netlist_header(f"r{i}", 27),
                f"V1 a 0 dc {i}",
                "R1 a 0 1k",
                op_control_block(["v(a)"]),
                END,
            ]
        )
        for i in range(12)
    ]
    res = run_many_cfg(decks, dev_cfg, expect=["v(a)"], progress=False)
    assert [r.values["v(a)"] for r in res] == pytest.approx(list(range(12)))


@requires_ngspice
def test_dataset_generation_is_deterministic(dev_cfg, card):
    cfg = replace(dev_cfg, sampling=replace(dev_cfg.sampling, n_samples_per_case=12))
    a = generate_case_dataset("pmos_off", cfg, card, save_example=False)
    b = generate_case_dataset("pmos_off", cfg, card, save_example=False)
    assert len(a) == 12
    pd.testing.assert_frame_equal(a, b)


@requires_ngspice
@pytest.mark.parametrize("gate_name", GATES)
def test_gate_subckt_logic(gate_name, dev_cfg, card):
    """Every gate subckt computes its truth table in SPICE and its internal nodes are probeable.

    Also a minimal working example of a gate deck for P3 (render_gate_netlist) and P4.
    """
    gate = get_gate(gate_name)
    vdd = dev_cfg.assignment.vdd_nominal
    sub = render_gate_subckt(gate, *gate_widths(dev_cfg), dev_cfg.assignment.lmin)
    probes = ["i(vvdd)", "v(out)", *[f"v(x1.{n})" for n in gate.internal_nodes]]
    for bits in GATE_INPUT_COMBOS[gate_name]:
        levels = input_levels(gate, bits)
        deck = "\n".join(
            [
                netlist_header(f"{gate_name} {bits}", 27.0),
                model_block(card, card.nominal["nmos"], card.nominal["pmos"]),
                sub,
                f"Vvdd vdd 0 dc {vdd}",
                *[f"V{pin} {pin} 0 dc {vdd * lv}" for pin, lv in levels.items()],
                f"X1 {' '.join(subckt_pins(gate)[:-2])} vdd 0 {gate_name}",
                op_control_block(probes),
                END,
            ]
        )
        res = run_netlist(deck, work_dir=dev_cfg.simulator.work_dir, expect=probes)
        assert res.ok, res.error
        expected = vdd * logic_function(gate_name, bits)
        assert abs(res.values["v(out)"] - expected) < 0.1 * vdd, (bits, res.values)
        assert -res.values["i(vvdd)"] > 0  # the supply delivers the leakage
        assert np.isfinite(list(res.values.values())).all()
