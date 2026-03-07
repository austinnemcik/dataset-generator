# dataset-generator

A FastAPI + SQLModel service for generating, ingesting, and exporting training datasets.

## Future plans
When exporting a dataset to jsonl, add an optional additional file for {dataset_name}summary.json that has metadata incl. examples, avg_prompt tokens, avg_completion_tokens, duplicates removed during lifecycle of dataset, label_distribution(correct/incorrect) once we add different types. 

Add hard negative generation
- Generate correct answer, incorrect answer, and partially correct answer labelled. 
Useful for DPO, ORPO, and RLHF training.

Track prompt patterns and enforce diversity.
Example metrics:
question types
instruction verbs
domains
Avoid datasets that look like:
Explain X
Explain Y
Explain Z
Instead ensure mix:
Explain
Summarize
Compare
Generate
Translate
Rewrite
Classify

Built in evaluation benchmarks we can use to test finetunin against.

Order training examples / datasets by difficulty and complexity. 


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
