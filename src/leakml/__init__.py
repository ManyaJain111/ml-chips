"""leakml: ML-driven PVT-variation-aware leakage estimation.

Package map (owners in brackets, see TODO.md):

    constants, config            shared contracts                       [P4 owns config loader]
    pdk, sampling                model card parsing, random draws       [P1]
    spice/                       ngspice runner + single-FET netlists   [P1]
    data/                        transistor datasets + data checks      [P1]
    plotting                     shared plot helpers                    [P1]
    ml/, metrics                 transistor ML models + error metrics   [P2]
    gates/                       Part B, gate-level SPICE vs ML         [P3]
    circuit/                     Part C, test circuit SPICE vs ML       [P4]

Contracts between packages are defined in SPEC.md. Change SPEC.md first, then the code.
"""

__version__ = "0.1.0"
