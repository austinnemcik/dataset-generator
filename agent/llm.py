import asyncio

from generics import TimedLabel, saveCost, timer

import logger

from .settings import DEFAULT_MODEL, calculate_price, client


def _summarize_for_log(text: str, max_chars: int = 600) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars]}... [truncated {len(cleaned) - max_chars} chars]"


async def run_agent_async(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    label: TimedLabel | None = None,
    run_id: str | None = None,
    dataset_key: str | None = None,
    topic: str | None = None,
    stage: str = "unknown",
):
    model = model or DEFAULT_MODEL
    if label:
        with timer(label):
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            except Exception as e:
                logger.saveToLog(f"[run_agent_async] completion call failed: {e}", "ERROR")
                raise RuntimeError("Model completion call failed") from e
    else:
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as e:
            logger.saveToLog(f"[run_agent_async] completion call failed: {e}", "ERROR")
            raise RuntimeError("Model completion call failed") from e
    total_cost, input_cost, output_cost = calculate_price(
        response.usage.prompt_tokens, response.usage.completion_tokens, model
    )
    logger.saveToLog(
        (
            "OpenRouter API call succeeded. "
            f"stage={stage} "
            f"model={model} "
            f"tokens={response.usage.total_tokens} "
            f"cost_usd={{total:{total_cost},input:{input_cost},output:{output_cost}}} "
            f"prompt_chars={len((system_prompt or '')) + len((user_prompt or ''))} "
            f"response_chars={len((response.choices[0].message.content or ''))} "
            f"response_preview={_summarize_for_log(response.choices[0].message.content, 180)}"
        ),
        "INFO",
    )
    if run_id:
        try:
            saveCost(
                run_id=run_id,
                dataset_key=dataset_key,
                topic=topic,
                model=model,
                stage=stage,
                usd_total=float(str(total_cost).replace("$", "")),
                prompt_tokens=int(response.usage.prompt_tokens),
                completion_tokens=int(response.usage.completion_tokens),
                total_tokens=int(response.usage.total_tokens),
            )
        except Exception as e:
            logger.saveToLog(f"[run_agent_async] Failed to write cost ledger: {e}", "ERROR")
    return response.choices[0].message.content

