import asyncio

from generics import TimedLabel, timer

import logger

from .settings import DEFAULT_MODEL, calculate_price, client


async def run_agent_async(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    label: TimedLabel | None = None,
):
    model = model or DEFAULT_MODEL
    print(f"Calling API with model: {model}")
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
    logger.saveToLog(
        f"Successfully called OpenRouter API.  System Prompt: {system_prompt}, User Prompt: {user_prompt}, Model Used: {model}, Tokens Used: {response.usage.total_tokens}, Total Cost: {calculate_price(response.usage.prompt_tokens, response.usage.completion_tokens, model)}  \n Response: {response.choices[0].message.content}",
        "INFO",
    )
    return response.choices[0].message.content

