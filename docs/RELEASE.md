# Release Checklist (Solo Dev)

- [ ] `alembic upgrade head` executed in target environment
- [ ] `make check` passes locally
- [ ] Required env vars present (`DATABASE_URL`, `OPENROUTER_API_KEY`)
- [ ] Export artifact path is writable (`EXPORT_ARTIFACT_DIR` if set)
- [ ] Runtime logs rotate as expected
- [ ] Last backup/export sanity checked
- [ ] Smoke test: `GET /` and one main workflow endpoint
