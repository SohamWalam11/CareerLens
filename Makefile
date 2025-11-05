PYTHON ?= python
PIP ?= pip
NPM ?= npm
DOCKER_COMPOSE ?= docker compose

install:
	$(PIP) install -r requirements.txt
	cd frontend && $(NPM) install

format:
	$(PYTHON) -m black backend
	$(PYTHON) -m ruff check backend --fix
	cd frontend && $(NPM) run format

lint:
	$(PYTHON) -m ruff check backend
	cd frontend && $(NPM) run lint

pytest:
	$(PYTHON) -m pytest backend

# required alias for CI scripts
test: pytest

migrate:
	cd backend && alembic upgrade head

 dev:
	$(DOCKER_COMPOSE) up --build

up:
	$(DOCKER_COMPOSE) up -d --build

down:
	$(DOCKER_COMPOSE) down --remove-orphans
