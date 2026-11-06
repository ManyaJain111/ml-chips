# Pinned environment for `make docker-run T=all` (Deliverable 9). Build: make docker-build
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ngspice make \
 && rm -rf /var/lib/apt/lists/*

# ngspice is OpenMP-threaded; busy-waiting threads of parallel ngspice processes starve each other.
ENV OMP_WAIT_POLICY=passive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv
WORKDIR /work
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src
RUN uv sync --frozen

# the repo is bind-mounted at /work by `make docker-run`; `make ... RUN=python` uses /opt/venv
CMD ["bash"]
