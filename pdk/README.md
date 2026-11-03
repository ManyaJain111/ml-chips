# pdk/

Put the **assigned** PTM model file here (e.g. `22nm_HP.pm`) and point
`configs/project.yaml → assignment.model_file` at it. Use the exact file the TAs gave the team.

The file should contain two BSIM4 (`level = 54`) `.model` blocks, `nmos` and `pmos`, which include
`vth0`, `toxe`, `ndep`, `u0` and `xj`. `scripts/01_extract_nominals.py` parses them into
`artifacts/nominals.json`.
