"""ngspice infrastructure.  Owner: P1.

    common.py       pieces every netlist shares (title, options, .temp, model block, .control/op)
    runner.py       run ngspice in batch mode, in parallel, and report failures
    parse.py        turn ngspice stdout into {name: value}
    fet_netlist.py  single-transistor netlists for Part A

Gate netlists live in leakml.gates.netlist (P3), circuit netlists in leakml.circuit.netlist (P4).
Both are built from common.py.
"""
