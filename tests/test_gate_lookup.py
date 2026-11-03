"""GateLeakageModel (interface P3 -> P4) and the bit-string-safe CSV helpers."""

import pandas as pd
import pytest

from leakml import io
from leakml.gates.ml_model import GateLeakageModel


def _table():
    rows = [
        {
            "gate": g,
            "inputs": b,
            "corner_id": c,
            "i_vdd_ml": 1e-9 * (c + 1) + i * 1e-12,
            "i_in_ml": 0.0,
        }
        for i, (g, b) in enumerate([("INV", "0"), ("INV", "1"), ("NAND2", "00"), ("NAND2", "01")])
        for c in range(3)
    ]
    return pd.DataFrame(rows)


def test_lookup():
    gm = GateLeakageModel.from_results(_table())
    assert gm.predict("NAND2", "01", 2) == pytest.approx(3e-9 + 3e-12)
    assert gm.corner_ids() == [0, 1, 2]
    with pytest.raises(KeyError, match="08_gate_ml"):
        gm.predict("NOR2", "01", 0)
    with pytest.raises(KeyError, match="metric"):
        gm.predict("INV", "0", 0, metric="i_bogus")


def test_restores_bits_mangled_by_plain_pandas(tmp_path):
    p = tmp_path / "gate_ml.csv"
    _table().to_csv(p, index=False)
    mangled = pd.read_csv(p)  # plain pandas: "01" -> 1, "00" -> 0
    assert mangled["inputs"].dtype.kind in "iu"
    assert GateLeakageModel.from_results(mangled).predict("NAND2", "01", 0) == pytest.approx(
        1e-9 + 3e-12
    )


def test_io_roundtrip_keeps_bit_strings(tmp_path):
    df = pd.DataFrame(
        {"vector": ["0010110", "0000001"], "inputs": ["01", "00"], "x": [1e-12, 2e-12]}
    )
    back = io.read_csv(io.write_csv(df, tmp_path / "t.csv"))
    pd.testing.assert_frame_equal(df, back)


def test_duplicates_rejected():
    t = pd.concat([_table(), _table().iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        GateLeakageModel.from_results(t)
