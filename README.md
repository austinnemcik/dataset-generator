# dataset-generator

A FastAPI + SQLModel service for generating, ingesting, and exporting training datasets.

## Local setup (quick start)

1. Use Python 3.12.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Set required env vars.
5. Run Alembic migrations.
6. Start the API.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/dev.txt
export DATABASE_URL="postgresql://user:pass@localhost:5432/dataset_generator"
export OPENROUTER_API_KEY="<your-key>"
make migrate-upgrade
make run
```

## Common developer commands

```bash
make lint
make test
make compile
make check
```

## Migration guardrail

In non-dev environments, apply migrations before launching the app:

```bash
alembic upgrade head
```

Do **not** rely on startup to patch schema drift.

## CI parity

CI runs:
- Ruff lint
- Pytest
- Python compile check

Keep local `make check` green before pushing to avoid drift from CI.
