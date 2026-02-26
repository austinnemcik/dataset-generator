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
from agent import AgentType,run_naming_agent,  get_embedding, cosine_similarity, is_duplicate, generate_dataset, save_responses, run_agent
class Example(BaseModel):
    instruction: str
    response: str

class IngestExamples(BaseModel):
    example: list[Example] | None = None
    prompt: str
    dataset_id: int
    dataset_description: str
    dataset_name: str

class Generation(BaseModel):
    agent: AgentType
    topic: str
    amount: int
    source_material: str | None = None # allow passing in source material to guide the dataset gen if applicable

class MergeRequest(BaseModel):
    dataset_ids: list[int]
    delete_originals: bool = False

data_router = APIRouter(prefix="/dataset", tags=["dataset"])

@data_router.post('/ingest')
def ingest_example(body: IngestExamples, session: Session = Depends(get_session)):
    dataset_name = body.dataset_name or RandomNameGenerator()
    dataset_description = body.dataset_description
    errors = 0
    example_amount = 0
    dataset = Dataset(
            name=dataset_name,
            description=dataset_description,
            examples = [])
    existing_embeddings = []
    if not (body.example): print("Bad body formatting")
    if not (body != IngestExamples): print("Bad body formatting EXAMPLE ")

    if (body.example):
        for ex in body.example:
            prompt = body.prompt
            instruction = ex.instruction
            response = ex.response
            if (len(instruction) < 2 or len(response) < 2 or len(prompt) < 2):
                errors += 1  # could be interesting to have an error list, add better errors then we can return it.  maybe in v2
                print("Bad example,  adding error")
                continue
            embedding = get_embedding(ex.instruction)
            embedding_str = json.dumps(embedding)  # save as json string to avoid pgvector for now
            if is_duplicate(embedding, existing_embeddings):
                errors += 1
                print("Example entry, skipping..")
                continue
            existing_embeddings.append(embedding)

            dataset.examples.append(TrainingExample
                (prompt=prompt, instruction=instruction,
                response=response, embedding=embedding_str))
            example_amount += 1
        if len(dataset.examples) < 1:
            return response_builder(success=False, message="No examples found!", errors=errors, statusCode=404)
        session.add(dataset)
        session.commit()
        return response_builder(success=True, message="Dataset succesfully parsed and saved to databse.", errors=errors, count=example_amount, statusCode=201) 
    return response_builder(success=False, message="An error occurred with the formatting of this example.", statusCode=400)
        
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
            "conversations": [
                {"from": "system", "value": "You are a helpful assistant."}, # need to save the prompt in the dataset,  then we can pull it out.  also need /delete endpoint
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


@data_router.post("/generate")
def get_dataset(body: Generation):

    agent_type = body.agent
    topic = body.topic
    amount = body.amount
    source_material = body.source_material
    dataset, prompt = generate_dataset(agent_type, topic, amount, source_material)
    save_responses(examples=dataset, prompt=prompt)
    return response_builder(success=True, message="Successfully generated dataset", statusCode=201)

@data_router.delete("/remove/{dataset_id}")
def delete_dataset(dataset_id: int, session: Session = Depends(get_session)):
    dataset = session.get(Dataset, dataset_id)
    session.delete(dataset)                                               # we should disable this endpoint by default so it doesn't get exposed since it's not behind auth,  although neither is the rest of it yet. maybe v2
    session.commit()
    return response_builder(success=True, message=f"Successfully removed Dataset {dataset.name}", statusCode=200)    

@data_router.post("/merge")
def merge_datasets(body: MergeRequest, session: Session = Depends(get_session)):
    errors = 0
    example_count = 0
    dataset_count = 0
    all_datasets: Dataset = []
    all_examples: TrainingExample = []
    for id in body.dataset_ids:
        dataset = session.get(Dataset, id)
        if not dataset: 
            errors += 1
            continue
        examples = session.exec(select(TrainingExample).where(TrainingExample.dataset_id == id)).all()
        dataset_count += 1
        all_datasets.append(dataset)
        example_count += len(examples)
        all_examples.extend(examples) # extend adds the whole list,  append just one item
    
    try:
        meta = run_naming_agent(all_examples)
    except: return response_builder(success=False, statusCode=400, message="An error occurred when calling the naming agent")


    dataset = Dataset(name=meta["name"], description=meta["description"], examples=all_examples)
    session.add(dataset)
    
    if(body.delete_originals == True):
        for ds in all_datasets:
            session.delete(ds)
        session.commit()
        return response_builder(success=True, statusCode=201, message=f"Successfully deleted {dataset_count} datasets with {example_count} examples and merged into 1 dataset.", errors=errors)
    session.commit()
    return response_builder(success=True, statusCode=201, message=f"Successfully added new dataset with {example_count} examples", errors=errors)


