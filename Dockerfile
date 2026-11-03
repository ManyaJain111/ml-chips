# Reproducible environment: pinned ngspice + Python deps.
# Use this for ALL final data generation, so every teammate's SPICE numbers come from the same
# simulator version (the report must state it; `make check` writes it to artifacts/env.json).
#
#   make docker-build
#   make docker-shell        # interactive shell, repo mounted at /work
#   make docker-run T=data   # run one Makefile target inside the container
#
FROM python:3.12-slim-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends ngspice make \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/

# The venv lives outside /work so that bind-mounting the repo does not hide it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    PATH="/opt/venv/bin:${PATH}" \
    MPLBACKEND=Agg

WORKDIR /work

# Dependency layer (cached unless pyproject/uv.lock change)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

# Project layer
COPY . .
RUN uv sync --frozen && chmod -R a+rX /opt/venv

CMD ["bash"]
