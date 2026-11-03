"""Part C: test-circuit validation (3-bit CLA or 8-bit parity checker).  Owner: P4.

definition.py  gate-level description of the circuits (the single source of truth)
logic.py       logic simulation: which inputs each gate sees for each input vector
netlist.py     expand the gate-level description into one transistor-level SPICE deck
spice_ref.py   SPICE reference: all input vectors x 200 corners
ml_model.py    hierarchical ML: sum of gate-model leakages
report.py      Deliverable 8
"""
