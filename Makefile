.PHONY: install install-dev lint test compile check run migrate-upgrade migrate-downgrade heads

install:
	pip install -r requirements/base.txt

install-dev:
	pip install -r requirements/dev.txt

lint:
	ruff check config.py database.py logger.py main.py routes/data_processing.py routes/dataset_data_routes.py services tests

test:
	pytest -q

compile:
	python -m py_compile config.py database.py logger.py main.py routes/data_processing.py routes/dataset_data_routes.py services/export_service.py services/import_service.py routes/dataset_models.py migrations/env.py migrations/versions/*.py

check: lint test compile

run:
	uvicorn main:app --host 127.0.0.1 --port 8000 --reload

migrate-upgrade:
	alembic upgrade head

migrate-downgrade:
	alembic downgrade -1

heads:
	alembic heads
