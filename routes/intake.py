from fastapi import APIRouter, File, UploadFile
import json
intake_router = APIRouter(prefix="/intake", tags=["intake"])

@intake_router.post("/upload")
async def ingest_file(file: UploadFile):
    contents = await file.read()

    if file.filename.endswith(".jsonl"):
        lines = contents.decode("utf-8").strip().split("\n")
        raw_examples = [json.loads(line) for line in lines if line.strip()]

