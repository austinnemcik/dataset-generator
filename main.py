from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, Response, Depends
from pydantic import BaseModel
from datetime import datetime
from typing import Literal, List
from sqlmodel import Session, select
from database import get_session, engine, Dataset, TrainingExample
from funkybob import RandomNameGenerator
import json
from generics import response_builder
app = FastAPI()

class Example(BaseModel):
    instruction: str
    response: str

class IngestExamples(BaseModel):
    raw: str | None = None
    example: list[Example] | None = None
    dataset_id: int
    dataset_description: str
    dataset_name: str

@app.get('/')
def get_root(): 
    return {"message": "Data Processing API"}

@app.post('/ingest')
def ingest_example(body: IngestExamples, session: Session = Depends(get_session)):
    dataset_name = body.dataset_name or RandomNameGenerator()
    dataset_description = body.dataset_description
    errors = 0
    examples = 0
    dataset = Dataset(
            name=dataset_name,
            description=dataset_description,
            examples = [])
    if (body.example):
        for ex in body.example:
            instruction = ex.instruction
            response = ex.response
            if (len(instruction) < 2 and len(response) < 2):
                errors += 1
                continue
            dataset.examples.append(TrainingExample
                (instruction=instruction,
                response=response))
            examples += 1
        if len(dataset.examples) < 1:
            return response_builder(success=True, message="No examples found!", errors=errors, statusCode=404)
        session.add(dataset)
        session.commit()
        return response_builder(success=True, message="Dataset succesfully parsed and saved to databse.", errors=errors, count=examples, statusCode=200) 
            
    if (body.raw):
        raw = body.raw
        pairs = raw.split("----") # use 4 to separate instruction / response pairs, --- to separate inside of that pair"
        # assume always an instruction and a response
        length = len(pairs)
       # if (length % 2 != 0):
          #  return JSONResponse(content = {
             #   "success": False,
             #   "message": "Straggler detected. Missing instruction / response pair"},
            # status_code=400)
        # but what if we have multiple missing pairs,  we coulda accidentally accept bad data
        # oh,  if we check pair by pair,  we can just reject that one data point instead of rejecting the entire file
        
        for pair in pairs:
            if (pair.strip() == ""):
                errors += 1
                continue
            parts = pair.strip().split("---", 1)
            if len(parts) < 2:
                errors += 1
                continue
            instruction = parts[0].strip()
            model_response = parts[1].strip()
            if len(instruction) < 2 or len(model_response) < 2:
                errors += 1
                continue # 
            dataset.examples.append(TrainingExample(
                instruction = instruction,
                response = model_response))
            examples += 1
        # once that loop is finished.  save
        if len(dataset.examples) == 0:
            return response_builder(success=True, message="No training examples found", errors=errors, statusCode=404)
        session.add(dataset)
        session.commit()
        return response_builder(success=True, message="Dataset successfully parsed and saved to database.", errors=errors, count=examples, statusCode=200)
        
@app.get("/datasets/{dataset_id}/export")
def export_dataset(dataset_id: int, session: Session = Depends(get_session)):
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        return response_builder(success=False, message="No dataset found for this ID", statusCode=404)
    
    examples = session.exec(
        select(TrainingExample).where(TrainingExample.dataset_id == dataset.id)
    ).all()

    lines = []
    for example in examples:
        formatted = {
            "conversation": [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": example.instruction},
                {"from": "gpt", "value": example.response}
            ]
        }
        lines.append(json.dumps(formatted))
    
    jsonl_content = "\n".join(lines)

    filepath = f"/tmp/{dataset.name}.jsonl"
    with open(filepath, "w", encoding='utf-8') as file:
        file.write(jsonl_content)

    return FileResponse(filepath, filename=f"{dataset.name}.jsonl")

@app.get("/datasets/amount/{dataset_amount}")
def all_datasets(dataset_amount: int, session: Session = Depends(get_session)):
    amount = dataset_amount
    if not dataset_amount:
        amount = 5
        # return 5 if we don't get an amount specified.
    datasets = session.exec(
    select(Dataset).order_by(Dataset.id).limit(amount)).all()

    details = []
    amt = 0
    for dataset in datasets:
        formatted = {
            "dataset": [
                {"name": dataset.name},
                {"description": dataset.description},
                {"id": dataset.id}
            ]
        }
        details.append(json.dumps(formatted))
        amt += 1

    return JSONResponse({
        "success": True,
        "message": f"Successfully returned {amt} datasets",
        "datasets": details
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)