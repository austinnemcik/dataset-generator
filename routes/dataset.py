from fastapi.responses import JSONResponse, FileResponse
from fastapi import Depends, APIRouter
from pydantic import BaseModel
from datetime import datetime
from sqlmodel import Session, select
from database import get_session, Dataset, TrainingExample
from funkybob import RandomNameGenerator
import json
import tempfile
import os
from generics import response_builder
from enum import Enum
from agent import generate_dataset, save_responses, qa_agent, instruction_following_agent, domain_specialist_agent,style_agent, adversarial_agent, conversation_agent, AgentType, agent_map
class Example(BaseModel):
    instruction: str
    response: str

class IngestExamples(BaseModel):
    raw: str | None = None
    example: list[Example] | None = None
    dataset_id: int
    dataset_description: str
    dataset_name: str

data_router = APIRouter(prefix="/dataset", tags=["dataset"])

@data_router.post('/ingest')
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
        
@data_router.get("/{dataset_id}/export")
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

    filepath = os.path.join(tempfile.gettempdir(), f"{dataset.name}.jsonl")
    with open(filepath, "w", encoding='utf-8') as file:
        file.write(jsonl_content)

    return FileResponse(filepath, filename=f"{dataset.name}.jsonl")

@data_router.get("/amount/{dataset_amount}")
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


class Generation(BaseModel):
    agent: AgentType
    topic: str
    amount: int
    source_material: str | None = None # allow passing in source material to guide the dataset gen if applicable

@data_router.post("/generate")
def get_dataset(body: Generation):
    agent_type = body.agent
    topic = body.topic
    amount = body.amount
    source_material = body.source_material
    dataset = generate_dataset(agent_type, topic, amount, source_material)
    save_responses(dataset)
    return response_builder(success=True, message="Successfully generated dataset", statusCode=201)