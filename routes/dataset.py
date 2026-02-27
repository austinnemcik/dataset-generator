from fastapi.responses import JSONResponse, FileResponse
from fastapi import Depends, APIRouter, Query
from pydantic import BaseModel
from datetime import datetime
from sqlmodel import Session, select
from database import get_session, Dataset, TrainingExample
from funkybob import RandomNameGenerator
import json
import tempfile
import os
from generics import response_builder, TimedLabel, timer, new_run_id
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
    model: str | None = None
    generation_cost: float | None = None
    grading_cost: float | None = None
    total_cost: float | None = None


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
        dataset_model = body.model
        generation_cost = float(body.generation_cost or 0.0)
        grading_cost = float(body.grading_cost or 0.0)
        total_cost = float(body.total_cost or 0.0)
        errors = 0
        example_amount = 0
        dataset = Dataset(
            name=dataset_name,
            description=dataset_description,
            model=dataset_model,
            generation_cost=generation_cost,
            grading_cost=grading_cost,
            total_cost=total_cost,
            examples=[],
        )
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
                {"model": dataset.model},
                {"generation_cost": dataset.generation_cost},
                {"grading_cost": dataset.grading_cost},
                {"total_cost": dataset.total_cost},
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


@data_router.get("/costs/summary")
def cost_summary(
    session: Session = Depends(get_session),
    limit: int | None = Query(default=None, ge=1, le=10000),
    model: str | None = Query(default=None),
):
    try:
        stmt = select(Dataset).order_by(Dataset.id)
        if model:
            stmt = stmt.where(Dataset.model == model)
        if limit:
            stmt = stmt.limit(limit)
        datasets = session.exec(stmt).all()
        overall = {
            "dataset_count": 0,
            "generation_cost": 0.0,
            "grading_cost": 0.0,
            "total_cost": 0.0,
        }
        by_model: dict[str, dict] = {}
        by_dataset: list[dict] = []

        for ds in datasets:
            model_key = ds.model or "unknown"
            overall["dataset_count"] += 1
            overall["generation_cost"] += float(ds.generation_cost or 0.0)
            overall["grading_cost"] += float(ds.grading_cost or 0.0)
            overall["total_cost"] += float(ds.total_cost or 0.0)

            if model_key not in by_model:
                by_model[model_key] = {
                    "dataset_count": 0,
                    "generation_cost": 0.0,
                    "grading_cost": 0.0,
                    "total_cost": 0.0,
                }
            by_model[model_key]["dataset_count"] += 1
            by_model[model_key]["generation_cost"] += float(ds.generation_cost or 0.0)
            by_model[model_key]["grading_cost"] += float(ds.grading_cost or 0.0)
            by_model[model_key]["total_cost"] += float(ds.total_cost or 0.0)

            by_dataset.append(
                {
                    "id": ds.id,
                    "name": ds.name,
                    "model": ds.model,
                    "generation_cost": round(float(ds.generation_cost or 0.0), 8),
                    "grading_cost": round(float(ds.grading_cost or 0.0), 8),
                    "total_cost": round(float(ds.total_cost or 0.0), 8),
                }
            )

        overall = {k: (round(v, 8) if isinstance(v, float) else v) for k, v in overall.items()}
        for model_key, vals in by_model.items():
            by_model[model_key] = {
                k: (round(v, 8) if isinstance(v, float) else v) for k, v in vals.items()
            }

        return JSONResponse(
            {
                "success": True,
                "message": "Successfully returned cost summary",
                "filters": {"model": model, "limit": limit},
                "overall": overall,
                "by_model": by_model,
                "by_dataset": by_dataset,
            }
        )
    except Exception as e:
        logger.saveToLog(f"[cost_summary] Unexpected error: {e}", "ERROR")
        return response_builder(
            success=False,
            message="An error occurred while building cost summary.",
            statusCode=500,
        )


@data_router.post("/generate")
async def get_dataset(body: Generation):
    agent_type = body.agent_type
    topic = body.topic
    amount = body.amount
    source_material = body.source_material
    model = body.model
    run_id = new_run_id()
    dataset_key = topic
    try:
        if body.model:
            dataset, prompt = await generate_dataset(
                agent_type=agent_type,
                topic=topic,
                amt=amount,
                source_material=source_material,
                model=model,
                run_id=run_id,
                dataset_key=dataset_key,
            )
        else:
            dataset, prompt = await generate_dataset(
                agent_type=agent_type,
                topic=topic,
                amt=amount,
                source_material=source_material,
                run_id=run_id,
                dataset_key=dataset_key,
            )
        await save_responses(
            agent_type=agent_type,
            examples=dataset,
            prompt=prompt,
            topic=topic,
            model=model,
            amount=amount,
            source_material=source_material,
            run_id=run_id,
            dataset_key=dataset_key,
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
