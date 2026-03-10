.PHONY: dev run down test lint format logs shell

# Build and start locally via Docker Compose (includes Postgres)
dev:
	docker compose up --build

# Run in detached mode
run:
	docker compose up --build -d

# Stop and remove containers
down:
	docker compose down

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
