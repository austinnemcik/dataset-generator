from sqlmodel import Session, select

from app.core.database import BatchRun, BatchRunItem, Dataset, engine, utcnow

from .batch_summary import build_batch_summary, json_dump, json_load, result_from_item, update_batch_counts


TERMINAL_ITEM_STATUSES = {"completed", "failed"}
BLOCKING_BATCH_STATUSES = {"paused", "cancelled"}


def load_existing_dataset_names(limit: int = 250) -> list[str]:
    with Session(engine) as session:
        datasets = session.exec(select(Dataset).order_by(Dataset.id.desc()).limit(limit)).all()
    names = [str(ds.name).strip() for ds in datasets if getattr(ds, "name", None)]
    return [name for name in names if name]


def create_batch_run(*, request_payload: dict, run_plan: list[dict], run_id: str) -> str:
    with Session(engine) as session:
        batch_run = BatchRun(
            run_id=run_id,
            status="queued",
            request_json=json_dump(request_payload),
            total_runs=len(run_plan),
            queued_runs=len(run_plan),
            running_runs=0,
            completed_runs=0,
            failed_runs=0,
        )
        session.add(batch_run)
        session.flush()

        for item in run_plan:
            session.add(
                BatchRunItem(
                    batch_run_id=batch_run.id,
                    item_index=int(item["item_index"]),
                    run_id=str(item["run_id"]),
                    dataset_key=str(item["dataset_key"]),
                    slot_key=item.get("slot_key"),
                    requested_topic=str(item["requested_topic"]),
                    topic=str(item["topic"]),
                    agent=str(item["agent"]),
                    ex_amt=int(item["ex_amt"]),
                    seed=item.get("seed"),
                    status="queued",
                    attempts=0,
                    max_retries=int(item["max_retries"]),
                    retry_backoff_seconds=float(item["retry_backoff_seconds"]),
                    source_material=item.get("source_material"),
                    model=item.get("model"),
                )
            )

        session.flush()
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        update_batch_counts(items, batch_run)
        build_batch_summary(batch_run, items)
        session.add(batch_run)
        session.commit()

    return run_id


def get_batch_run_status(batch_run_id: str) -> dict | None:
    with Session(engine) as session:
        batch_run = session.exec(select(BatchRun).where(BatchRun.run_id == batch_run_id)).first()
        if not batch_run:
            return None
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        update_batch_counts(items, batch_run)
        summary = build_batch_summary(batch_run, items)
        session.add(batch_run)
        session.commit()
        return summary


def pause_batch_run(batch_run_id: str) -> dict | None:
    with Session(engine) as session:
        batch_run = session.exec(select(BatchRun).where(BatchRun.run_id == batch_run_id)).first()
        if not batch_run:
            return None
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        if batch_run.status in {"completed", "failed", "cancelled"}:
            return build_batch_summary(batch_run, items)
        batch_run.status = "paused"
        batch_run.updated_at = utcnow()
        build_batch_summary(batch_run, items)
        session.add(batch_run)
        session.commit()
        return build_batch_summary(batch_run, items)


def stop_batch_run(batch_run_id: str) -> dict | None:
    with Session(engine) as session:
        batch_run = session.exec(select(BatchRun).where(BatchRun.run_id == batch_run_id)).first()
        if not batch_run:
            return None
        now = utcnow()
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        for item in items:
            if item.status == "queued":
                item.status = "failed"
                item.error_type = "cancelled"
                item.error = "Stopped by user before execution"
                item.completed_at = now
                item.updated_at = now
                item.result_json = json_dump(result_from_item(item))
            elif item.status == "running":
                item.error_type = "cancelled"
                item.error = "Stop requested while item was already running"
                item.updated_at = now
        batch_run.status = "cancelled"
        batch_run.updated_at = now
        if batch_run.running_runs == 0:
            batch_run.completed_at = now
        update_batch_counts(items, batch_run)
        summary = build_batch_summary(batch_run, items)
        session.add(batch_run)
        session.commit()
        return summary


def delete_batch_run(batch_run_id: str) -> dict | None:
    with Session(engine) as session:
        batch_run = session.exec(select(BatchRun).where(BatchRun.run_id == batch_run_id)).first()
        if not batch_run:
            return None
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        if batch_run.status in {"queued", "running", "paused"}:
            raise ValueError("Only completed, failed, or cancelled batch runs can be removed.")
        summary = build_batch_summary(batch_run, items)
        session.delete(batch_run)
        session.commit()
        return summary


def delete_terminal_batch_runs() -> dict:
    with Session(engine) as session:
        runs = session.exec(
            select(BatchRun)
            .where(BatchRun.status.in_(["completed", "failed", "cancelled"]))
            .order_by(BatchRun.created_at)
        ).all()
        if not runs:
            return {"deleted_run_ids": [], "deleted_count": 0}

        deleted_run_ids: list[str] = []
        for batch_run in runs:
            deleted_run_ids.append(batch_run.run_id)
            session.delete(batch_run)

        session.commit()
        return {"deleted_run_ids": deleted_run_ids, "deleted_count": len(deleted_run_ids)}


def restart_failed_batch_run(batch_run_id: str) -> dict | None:
    with Session(engine) as session:
        batch_run = session.exec(select(BatchRun).where(BatchRun.run_id == batch_run_id)).first()
        if not batch_run:
            return None

        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        failed_items = [item for item in items if item.status == "failed"]
        if not failed_items:
            return build_batch_summary(batch_run, items)

        now = utcnow()
        for item in failed_items:
            item.status = "queued"
            item.attempts = 0
            item.error_type = None
            item.error = None
            item.created_dataset_id = None
            item.result_json = None
            item.started_at = None
            item.completed_at = None
            item.updated_at = now

        if batch_run.status in {"failed", "completed", "cancelled", "paused"}:
            batch_run.status = "queued"
        batch_run.completed_at = None
        batch_run.updated_at = now
        update_batch_counts(items, batch_run)
        summary = build_batch_summary(batch_run, items)
        session.add(batch_run)
        session.commit()
        return summary


def batch_run_control_state(item_id: int) -> tuple[str | None, str] | None:
    with Session(engine) as session:
        item = session.get(BatchRunItem, item_id)
        if not item:
            return None
        batch_run = session.get(BatchRun, item.batch_run_id)
        if not batch_run:
            return None
        return batch_run.run_id, batch_run.status


def claim_existing_dataset_if_present(item_id: int) -> dict | None:
    with Session(engine) as session:
        item = session.get(BatchRunItem, item_id)
        if not item:
            return None
        batch_run = session.get(BatchRun, item.batch_run_id)
        existing_dataset = session.exec(
            select(Dataset).where(Dataset.source_run_id == item.run_id)
        ).first()
        if not existing_dataset:
            return None
        item.status = "completed"
        item.created_dataset_id = existing_dataset.id
        item.error = None
        item.error_type = None
        item.completed_at = item.completed_at or utcnow()
        item.updated_at = utcnow()
        item.result_json = json_dump(result_from_item(item))
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        update_batch_counts(items, batch_run)
        build_batch_summary(batch_run, items)
        session.add(item)
        session.add(batch_run)
        session.commit()
        return result_from_item(item)


def prepare_item_attempt(item_id: int) -> tuple[dict, bool] | None:
    with Session(engine) as session:
        item = session.get(BatchRunItem, item_id)
        if not item:
            return None
        batch_run = session.get(BatchRun, item.batch_run_id)
        request_payload = json_load(batch_run.request_json, {})
        if item.status in TERMINAL_ITEM_STATUSES:
            return None
        if batch_run.status in BLOCKING_BATCH_STATUSES:
            return None

        item.attempts += 1
        item.status = "running"
        item.started_at = item.started_at or utcnow()
        item.updated_at = utcnow()
        item.error = None
        item.error_type = None
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        update_batch_counts(items, batch_run)
        build_batch_summary(batch_run, items)
        session.add(item)
        session.add(batch_run)
        session.commit()
        session.refresh(item)
        return (
            {
                "id": item.id,
                "batch_run_id": batch_run.run_id,
                "item_index": item.item_index,
                "run_id": item.run_id,
                "dataset_key": item.dataset_key,
                "requested_topic": item.requested_topic,
                "topic": item.topic,
                "agent": item.agent,
                "ex_amt": item.ex_amt,
                "attempts": item.attempts,
                "max_retries": item.max_retries,
                "retry_backoff_seconds": item.retry_backoff_seconds,
                "source_material": item.source_material,
                "source_material_mode": request_payload.get("source_material_mode", "content_and_style"),
                "conversation_length_mode": request_payload.get("conversation_length_mode", "varied"),
                "model": item.model,
                "slot_key": item.slot_key,
                "seed": item.seed,
            },
            item.attempts > (item.max_retries + 1),
        )


def finalize_item_success(item_id: int, dataset_id: int | None = None) -> dict | None:
    with Session(engine) as session:
        item = session.get(BatchRunItem, item_id)
        if not item:
            return None
        batch_run = session.get(BatchRun, item.batch_run_id)
        item.status = "completed"
        item.created_dataset_id = dataset_id
        item.error = None
        item.error_type = None
        item.completed_at = utcnow()
        item.updated_at = utcnow()
        item.result_json = json_dump(result_from_item(item))
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        update_batch_counts(items, batch_run)
        build_batch_summary(batch_run, items)
        session.add(item)
        session.add(batch_run)
        session.commit()
        return result_from_item(item)


def finalize_item_failure(item_id: int, *, error_type: str, error: str, terminal: bool) -> dict | None:
    with Session(engine) as session:
        item = session.get(BatchRunItem, item_id)
        if not item:
            return None
        batch_run = session.get(BatchRun, item.batch_run_id)
        item.error_type = error_type
        item.error = error
        item.updated_at = utcnow()
        if terminal:
            item.status = "failed"
            item.completed_at = utcnow()
        else:
            item.status = "queued"
        item.result_json = json_dump(result_from_item(item))
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        update_batch_counts(items, batch_run)
        build_batch_summary(batch_run, items)
        session.add(item)
        session.add(batch_run)
        session.commit()
        return result_from_item(item)


def list_incomplete_batch_runs() -> list[str]:
    with Session(engine) as session:
        runs = session.exec(
            select(BatchRun).where(BatchRun.status.in_(["queued", "running"])).order_by(BatchRun.created_at)
        ).all()
        return [run.run_id for run in runs]


def pending_item_ids_for_batch(batch_run_id: str) -> tuple[dict, list[int]]:
    with Session(engine) as session:
        batch_run = session.exec(select(BatchRun).where(BatchRun.run_id == batch_run_id)).first()
        if not batch_run:
            raise ValueError("Batch run not found")
        if batch_run.status == "cancelled":
            items = session.exec(
                select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
            ).all()
            return build_batch_summary(batch_run, items), []
        if batch_run.status == "paused":
            batch_run.status = "running"

        request_payload = json_load(batch_run.request_json, {})
        items = session.exec(
            select(BatchRunItem).where(BatchRunItem.batch_run_id == batch_run.id)
        ).all()
        for item in items:
            if item.status == "running":
                item.status = "queued"
                item.updated_at = utcnow()
        update_batch_counts(items, batch_run)
        build_batch_summary(batch_run, items)
        session.add(batch_run)
        session.commit()

        pending_item_ids = [item.id for item in items if item.status not in TERMINAL_ITEM_STATUSES]
        return request_payload, pending_item_ids



