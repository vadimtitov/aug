.PHONY: run down rebuild test lint format check logs shell

# Build and start — streams aug + postgres logs, suppresses chromium noise
run:
	docker compose up --build -d && docker compose logs -f aug postgres chromium

# Stop and remove containers
down:
	docker compose down

# Full rebuild — stop, rebuild images, start
rebuild:
	docker compose down && docker compose up --build -d && docker compose logs -f aug postgres chromium

# Run lint, format check, and tests
check: lint
	.venv/bin/ruff format --check .
	.venv/bin/pytest

# Run the test suite
test:
	.venv/bin/pytest

# Check for linting issues
lint:
	.venv/bin/ruff check .

# Auto-fix formatting and import order
format:
	.venv/bin/ruff format .

# Tail logs from the aug container
logs:
	docker compose logs -f aug

# Open a shell inside the running aug container
shell:
	docker compose exec aug /bin/bash

# Scenario suite against fake Telegram + fake LLM (free; see scripts/scenarios.py).
# Writes .scenarios/transcript-<commit>.txt; diff it against another commit's file.
SCENARIOS_COMPOSE = docker compose -f docker-compose.scenarios.yml -p aug-scenarios
ID_FIELD ?= context_id
scenarios:
	$(SCENARIOS_COMPOSE) down -v --remove-orphans
	.venv/bin/python scripts/scenarios.py seed .scenarios/data
	$(SCENARIOS_COMPOSE) up -d --build
	.venv/bin/python scripts/scenarios.py run --id-field $(ID_FIELD) --out .scenarios/transcript-$$(git rev-parse --short HEAD).txt $(if $(ONLY),--only $(ONLY)); \
	  status=$$?; $(SCENARIOS_COMPOSE) logs aug > .scenarios/aug.log 2>&1; $(SCENARIOS_COMPOSE) down -v; exit $$status
