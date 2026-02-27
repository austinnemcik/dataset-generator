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
from generics import response_builder, TimedLabel, timer
from agent import (
    AgentType,
    run_naming_agent,
    get_embedding,
    cosine_similarity,
    is_duplicate,
    generate_dataset,
    save_responses,
)
import logger


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
    agent_type: AgentType
    topic: str
    amount: int
    source_material: str | None = None  # allow passing in source material to guide the dataset gen if applicable
    model: str | None = None


class MergeRequest(BaseModel):
    dataset_ids: list[int]
    delete_originals: bool = False


data_router = APIRouter(prefix="/dataset", tags=["dataset"])


@data_router.post("/ingest")
def ingest_example(body: IngestExamples, session: Session = Depends(get_session)):
    try:
        dataset_name = body.dataset_name or str(RandomNameGenerator())
        dataset_description = body.dataset_description
        errors = 0
        example_amount = 0
        dataset = Dataset(name=dataset_name, description=dataset_description, examples=[])
        existing_embeddings = []
        if not body.example:
            raise ValueError("No examples provided in ingest payload")

        with timer(label=TimedLabel.INGEST_REQUEST):
            for ex in body.example:
                prompt = body.prompt
                instruction = ex.instruction
                response = ex.response
                if len(instruction) < 2 or len(response) < 2 or len(prompt) < 2:
                    errors += 1
                    logger.saveToLog(
                        "Discarding example with reason: BAD RESPONSE.. continuing",
                        "WARNING",
                    )
                    continue
                embedding = get_embedding(ex.instruction)
                embedding_str = json.dumps(embedding)
                if is_duplicate(embedding, existing_embeddings):
                    errors += 1
                    continue
                existing_embeddings.append(embedding)
                dataset.examples.append(
                    TrainingExample(
                        prompt=prompt,
                        instruction=instruction,
                        response=response,
                        embedding=embedding_str,
                    )
                )
                example_amount += 1

        if len(dataset.examples) < 1:
            raise ValueError("No valid examples found after ingest validation")

        session.add(dataset)
        session.commit()
        return response_builder(
            success=True,
            message="Dataset successfully parsed and saved to database.",
            errors=errors,
            count=example_amount,
            statusCode=201,
        )
    except ValueError as e:
        logger.saveToLog(f"[ingest_example] Validation failed: {e}", "ERROR")
        return response_builder(
            success=False,
            message=str(e),
            statusCode=400,
        )
    except Exception as e:
        logger.saveToLog(f"[ingest_example] Unexpected error: {e}", "ERROR")
        return response_builder(
            success=False,
            message="An error occurred while ingesting examples.",
            statusCode=500,
        )


@data_router.get("/{dataset_id}/export")
def export_dataset(dataset_id: int, session: Session = Depends(get_session)):
    try:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise ValueError("No dataset found for this ID")

        examples = session.exec(
            select(TrainingExample).where(TrainingExample.dataset_id == dataset.id)
        ).all()

        lines = []
        for example in examples:
            formatted = {
                "conversations": [
                    {"from": "system", "value": "You are a helpful assistant."},
                    {"from": "human", "value": example.instruction},
                    {"from": "gpt", "value": example.response},
                ]
            }
            lines.append(json.dumps(formatted))

        jsonl_content = "\n".join(lines)
        filepath = os.path.join(tempfile.gettempdir(), f"{dataset.name}.jsonl")
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(jsonl_content)
        return FileResponse(filepath, filename=f"{dataset.name}.jsonl")
    except ValueError as e:
        logger.saveToLog(f"[export_dataset] Validation failed: {e}", "ERROR")
        return response_builder(success=False, message=str(e), statusCode=404)
    except Exception as e:
        logger.saveToLog(f"[export_dataset] Unexpected error: {e}", "ERROR")
        return response_builder(
            success=False,
            message="An error occurred while exporting dataset.",
            statusCode=500,
        )


@data_router.get("/amount/{dataset_amount}")
def all_datasets(dataset_amount: int, session: Session = Depends(get_session)):
    amount = dataset_amount
    if not dataset_amount:
        amount = 5
        # return 5 if we don't get an amount specified.
    datasets = session.exec(select(Dataset).order_by(Dataset.id).limit(amount)).all()

    details = []
    amt = 0
    for dataset in datasets:
        formatted = {
            "dataset": [
                {"name": dataset.name},
                {"description": dataset.description},
                {"id": dataset.id},
            ]
        }
        details.append(json.dumps(formatted))
        amt += 1

    return JSONResponse(
        {
            "success": True,
            "message": f"Successfully returned {amt} datasets",
            "datasets": details,
        }
    )


@data_router.post("/generate")
async def get_dataset(body: Generation):
    agent_type = body.agent_type
    topic = body.topic
    amount = body.amount
    source_material = body.source_material
    model = body.model
    try:
        if body.model:
            dataset, prompt = await generate_dataset(
                agent_type=agent_type,
                topic=topic,
                amt=amount,
                source_material=source_material,
                model=model,
            )
        else:
            dataset, prompt = await generate_dataset(
                agent_type=agent_type,
                topic=topic,
                amt=amount,
                source_material=source_material,
            )
        await save_responses(
            agent_type=agent_type,
            examples=dataset,
            prompt=prompt,
            topic=topic,
            model=model,
            amount=amount,
            source_material=source_material,
        )
        return response_builder(
            success=True, message="Successfully generated dataset", statusCode=201
        )
    except ValueError as e:
        logger.saveToLog(f"[get_dataset] Validation failed: {e}", "ERROR")
        return response_builder(
            success=False,
            message=str(e),
            statusCode=400,
        )
    except Exception as e:
        logger.saveToLog(f"[get_dataset] Unexpected generation error: {e}", "ERROR")
        return response_builder(
            success=False,
            message="An unexpected error occurred while generating dataset.",
            statusCode=500,
        )


@data_router.delete("/remove/{dataset_id}")
def delete_dataset(dataset_id: int, session: Session = Depends(get_session)):
    try:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise ValueError("Dataset not found")
        session.delete(dataset)
        session.commit()
        return response_builder(
            success=True,
            message=f"Successfully removed Dataset {dataset.name}",
            statusCode=200,
        )
    except ValueError as e:
        logger.saveToLog(f"[delete_dataset] Validation failed: {e}", "ERROR")
        return response_builder(success=False, message=str(e), statusCode=404)
    except Exception as e:
        logger.saveToLog(f"[delete_dataset] Unexpected error: {e}", "ERROR")
        return response_builder(
            success=False,
            message="An error occurred while removing dataset.",
            statusCode=500,
        )


@data_router.post("/merge")
async def merge_datasets(body: MergeRequest, session: Session = Depends(get_session)):
    errors = 0
    example_count = 0
    dataset_count = 0
    all_datasets: Dataset = []
    all_examples: TrainingExample = []
    try:
        for id in body.dataset_ids:
            dataset = session.get(Dataset, id)
            if not dataset:
                errors += 1
                continue
            examples = session.exec(
                select(TrainingExample).where(TrainingExample.dataset_id == id)
            ).all()
            dataset_count += 1
            all_datasets.append(dataset)
            example_count += len(examples)
            all_examples.extend(examples)

        if not all_examples:
            raise ValueError("No examples found to merge")

        meta = await run_naming_agent(all_examples)
        if not meta or "name" not in meta or "description" not in meta:
            raise ValueError("Naming agent did not return valid metadata")

        dataset = Dataset(
            name=meta["name"], description=meta["description"], examples=all_examples
        )
        session.add(dataset)

        if body.delete_originals is True:
            for ds in all_datasets:
                session.delete(ds)
            session.commit()
            return response_builder(
                success=True,
                statusCode=201,
                message=f"Successfully deleted {dataset_count} datasets with {example_count} examples and merged into 1 dataset.",
                errors=errors,
            )
        session.commit()
        return response_builder(
            success=True,
            statusCode=201,
            message=f"Successfully added new dataset with {example_count} examples",
            errors=errors,
        )
    except ValueError as e:
        logger.saveToLog(f"[merge_datasets] Validation failed: {e}", "ERROR")
        return response_builder(success=False, statusCode=400, message=str(e))
    except Exception as e:
        logger.saveToLog(f"[merge_datasets] Unexpected error: {e}", "ERROR")
        return response_builder(
            success=False,
            statusCode=500,
            message="An unexpected error occurred while merging datasets",
        )
