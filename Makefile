# ABIDES — student-facing convenience targets.
#
# All targets work the same on Mac and Windows (Windows users: install GNU make
# via `winget install GnuWin32.Make`, or run the underlying `docker compose ...`
# command directly — see each recipe).
#
# Common flow:
#   make build     # one-time: build the Docker image
#   make smoke     # confirm the environment works
#   make up        # start JupyterLab on http://localhost:8888
#   make shell     # drop into a shell inside the container
#   make test      # run the pytest suite
#   make down      # stop the container
#   make clean     # remove the image and volumes (full reset)

# Use bash for recipes (better error handling than /bin/sh).
SHELL := /usr/bin/env bash

DC := docker compose

.DEFAULT_GOAL := help

.PHONY: help build up down logs shell test smoke tournament clean rebuild lock

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?##"} {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

build:  ## Build the abides-lab image.
	$(DC) build

rebuild:  ## Rebuild from scratch (no cache).
	$(DC) build --no-cache

up:  ## Start JupyterLab on http://localhost:8888.
	$(DC) up

down:  ## Stop and remove the container.
	$(DC) down

logs:  ## Tail container logs.
	$(DC) logs -f

shell:  ## Drop into a bash shell inside the container.
	$(DC) run --rm --service-ports abides bash

test:  ## Run the pytest suite inside the container.
	$(DC) run --rm abides pytest -q

smoke:  ## Run the 5-minute RMSC04 smoke test inside the container.
	$(DC) run --rm abides python tools/smoke_test.py

tournament:  ## Run the class tournament (full day). Override: make tournament SEED=123 END_TIME=10:00:00
	$(DC) run --rm abides python tools/run_tournament.py \
		--seed $(or $(SEED),42) \
		--end-time $(or $(END_TIME),17:30:00)

lock:  ## Regenerate requirements.lock from the *.in files. Commit the result.
	$(DC) run --rm abides \
		uv pip compile \
			--quiet \
			--output-file=requirements.lock \
			requirements.in requirements-dev.in
	@echo "requirements.lock regenerated. Review the diff and commit it."

clean:  ## Remove image and dangling containers (full reset).
	-$(DC) down --rmi local --volumes --remove-orphans
