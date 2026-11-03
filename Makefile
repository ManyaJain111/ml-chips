# Pipeline entry points. Every step is a numbered script in scripts/ (Deliverable 9:
# "scripts that regenerate everything from the model file, run in order, with fixed seeds").
#
#   make setup         # uv sync
#   make all           # full pipeline, Part A -> B -> C
#   make partA QUICK=1 # pilot run with small sample counts (writes to data/_pilot, results/_pilot)
#
# Steps depend on each other's outputs: never run targets in parallel, even if MAKEFLAGS has -j.
# (Each script already parallelises its own ngspice runs.)
.NOTPARALLEL:

CONFIG ?= configs/project.yaml
RUN    ?= uv run python
QUICK  ?=
QFLAG   = $(if $(QUICK),--quick,)
ARGS    = --config $(CONFIG) $(QFLAG)

IMAGE  ?= leakml:latest
T      ?= all

.PHONY: setup check nominals data datacheck train finalize corners gates-spice gates-ml \
        gates-report circuit-spice circuit-ml circuit-report partA partB partC all baseline bootstrap \
        test lint fmt docker-build docker-shell docker-run

setup:
	uv sync

# ---------------------------------------------------------------- Part A (P1, P2)
check:          ; $(RUN) scripts/00_check_env.py $(ARGS)
nominals:       ; $(RUN) scripts/01_extract_nominals.py $(ARGS)
data:           ; $(RUN) scripts/02_generate_transistor_data.py $(ARGS)
datacheck:      ; $(RUN) scripts/03_data_check.py $(ARGS)
train:          ; $(RUN) scripts/04_train_compare.py $(ARGS)
finalize:       ; $(RUN) scripts/05_finalize_models.py $(ARGS)

# ---------------------------------------------------------------- Part B (P3)
corners:        ; $(RUN) scripts/06_generate_corners.py $(ARGS)
gates-spice:    ; $(RUN) scripts/07_gate_spice.py $(ARGS)
gates-ml:       ; $(RUN) scripts/08_gate_ml.py $(ARGS)
gates-report:   ; $(RUN) scripts/09_gate_report.py $(ARGS)

# ---------------------------------------------------------------- Part C (P4)
circuit-spice:  ; $(RUN) scripts/10_circuit_spice.py $(ARGS)
circuit-ml:     ; $(RUN) scripts/11_circuit_ml.py $(ARGS)
circuit-report: ; $(RUN) scripts/12_circuit_report.py $(ARGS)

partA: check nominals data datacheck train finalize
partB: corners gates-spice gates-ml gates-report
partC: circuit-spice circuit-ml circuit-report
all:   partA partB partC

# ---------------------------------------------------------------- independent development
# Everything P2/P3/P4 need to start without waiting for each other (~1 min with QUICK=1):
#   pilot transistor data, shared corners, STAND-IN transistor models.
#   make bootstrap CONFIG=configs/dev.yaml QUICK=1
baseline:       ; $(RUN) scripts/dev_train_baseline.py $(ARGS)
bootstrap: check nominals data corners baseline

# ---------------------------------------------------------------- dev
test: ; uv run pytest
lint: ; uv run ruff check src scripts tests
fmt:  ; uv run ruff format src scripts tests

# ---------------------------------------------------------------- container
docker-build:
	docker build -t $(IMAGE) .

docker-shell:
	docker run --rm -it --user $$(id -u):$$(id -g) -v $$PWD:/work $(IMAGE) bash

docker-run:
	docker run --rm --user $$(id -u):$$(id -g) -v $$PWD:/work $(IMAGE) make $(T) RUN=python CONFIG=$(CONFIG) QUICK=$(QUICK)
