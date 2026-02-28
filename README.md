# dataset-generator
A small utility that can be used to generate, ingest, and export formatted datasets using Python and Postgres. Built on top of the OpenAI Agents SDK.

## Intake UI reference card
Use `GET /dataset/intake/reference` to retrieve a reference card for the scraper intake endpoint (`POST /dataset/intake/scraper`), including a ready-to-run curl example and payload shape guidance.

For larger scraper dumps, send records in chunks and tune `max_records`, `chunk_size`, `dedupe_against_existing`, and `dedupe_within_payload` for throughput.
