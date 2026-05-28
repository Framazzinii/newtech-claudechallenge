# ABIDES — Linux container for the "Build and Battle Your Trading Agent" lab.
#
# Designed to give every student (Mac Intel, Apple Silicon, or Windows) the
# same Python 3.11 environment without "works on my machine" issues.
#
# Two ways to use this image:
#   1. VS Code "Reopen in Container" (.devcontainer/devcontainer.json)
#      → drops you into a shell with the source bind-mounted in editable mode.
#   2. `docker compose up`
#      → starts JupyterLab on http://localhost:8888.
#
# Phase 3 changes:
#   - Python 3.9 → 3.11.
#   - Sub-packages declare their deps in pyproject.toml (PEP 621), not setup.cfg.
#   - Installs use `uv` instead of `pip` for ~5–10× faster builds, especially
#     noticeable on Windows. Lockfile generation also uses `uv pip compile`.

FROM python:3.11-slim-bookworm

# --- Install uv (fast pip replacement) --------------------------------------
# Official multi-stage copy from the upstream uv image is the documented
# fastest path; no curl, no pip bootstrap, no extra layers.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# --- System-level build deps -------------------------------------------------
# Needed by numba/llvmlite, pomegranate (no arm64 wheels), etc.
# `make` and `git` are conveniences for in-container workflows.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        make \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Non-root user -----------------------------------------------------------
# Linux hosts mount bind volumes with the host UID; running as root would
# create files students can't edit from their host shell. Docker Desktop on
# Mac/Windows handles UID mapping transparently, so this is a Linux-host
# courtesy.
ARG USERNAME=abides
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd  --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME}

WORKDIR /abides

# --- Python dependencies -----------------------------------------------------
# Two paths:
#   1. requirements.lock present → fully-pinned, reproducible install.
#   2. requirements.lock absent  → bootstrap from the .in files. Used when
#      first cloning the repo before `make lock` has ever run.
#
# uv pip install --system installs into the container's global Python (no
# virtualenv needed since the container itself is the isolation boundary).
COPY requirements.in requirements-dev.in ./
COPY requirements.lock* ./

# Sub-package metadata first so the heavy pip layer caches well.
COPY abides-core/pyproject.toml    abides-core/pyproject.toml
COPY abides-markets/pyproject.toml abides-markets/pyproject.toml
COPY abides-gym/pyproject.toml     abides-gym/pyproject.toml

# Repo-level pyproject.toml (tool config only — not a package).
COPY pyproject.toml ./pyproject.toml

RUN if [ -f requirements.lock ]; then \
        echo "Installing from requirements.lock (reproducible)" && \
        uv pip install --system --no-cache -r requirements.lock ; \
    else \
        echo "No requirements.lock found — bootstrapping from .in files" && \
        uv pip install --system --no-cache -r requirements.in -r requirements-dev.in ; \
    fi

# --- ABIDES sub-packages (editable) -----------------------------------------
# Copy each sub-package's source tree and install in editable mode so source
# edits in the bind-mounted /abides take effect without a rebuild.
COPY abides-core/    abides-core/
COPY abides-markets/ abides-markets/
COPY abides-gym/     abides-gym/

# In the lockfile path, all deps were already installed above; here uv just
# registers the editable links and confirms install_requires is satisfied.
# In the bootstrap path, the editable installs pull each sub-package's deps.
RUN uv pip install --system --no-cache -e ./abides-core \
 && uv pip install --system --no-cache -e ./abides-markets \
 && uv pip install --system --no-cache -e ./abides-gym

# Copy the rest of the tree (notebooks, tests, tools, etc.).
# The bind-mount at runtime will overlay this anyway, but copying makes the
# image self-contained when run without a mount.
COPY . .

# Directory for student notebooks / output (typically bind-mounted).
RUN mkdir -p /abides/student_work \
 && chown -R ${USER_UID}:${USER_GID} /abides

USER ${USERNAME}

EXPOSE 8888

# JupyterLab on 8888. JUPYTER_TOKEN is empty by default for class
# convenience; set it in compose / shell env if you ever expose the port
# beyond localhost.
ENV JUPYTER_TOKEN=""
CMD ["sh", "-c", "jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --notebook-dir=/abides \
        --ServerApp.token=${JUPYTER_TOKEN} \
        --ServerApp.password="]
