.DEFAULT_GOAL := help
COMPOSE := docker compose

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## First-time setup: copy .env and build images
	@test -f .env || (cp .env.example .env && echo "Created .env -- add your GOOGLE_API_KEY")
	$(COMPOSE) build

up: ## Start the whole stack
	$(COMPOSE) up -d
	@echo "App:  http://localhost:5173"
	@echo "API:  http://localhost:8000/docs"

down: ## Stop the stack
	$(COMPOSE) down

logs: ## Tail API logs
	$(COMPOSE) logs -f api

ps: ## Show container status
	$(COMPOSE) ps

shell: ## Open a shell in the API container
	$(COMPOSE) exec api bash

migrate: ## Apply migrations
	$(COMPOSE) exec api alembic upgrade head

revision: ## Autogenerate a migration: make revision m="add foo"
	$(COMPOSE) exec api alembic revision --autogenerate -m "$(m)"

test: ## Run the test suite
	$(COMPOSE) exec api pytest

lint: ## Ruff check + format
	$(COMPOSE) exec api ruff check --fix app tests
	$(COMPOSE) exec api ruff format app tests

clean: ## Stop and delete volumes (destroys local data)
	$(COMPOSE) down -v

.PHONY: help setup up down logs ps shell migrate revision test lint clean
