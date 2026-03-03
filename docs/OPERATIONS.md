# Operations Notes

## Log hygiene

- Logs rotate via `RotatingFileHandler` in `logger.py`.
- Runtime logs are gitignored.
- Optional cleanup: periodically archive/remove stale export artifacts.

## Failure observability (lightweight)

Track these counts from logs or DB tables:

- Import failures (`ImportHistory.status == "failed"`)
- Export failures (`ExportHistory.status == "failed"`)
- Batch run failures (`BatchRun.status == "failed"`)

Suggested weekly check:

1. Query latest 7 days for each failed status.
2. Scan top repeating errors.
3. Fix one recurring failure source per week.
