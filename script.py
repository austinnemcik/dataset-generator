import asyncio
import httpx

SERVER_URL = "http://localhost:8000"

MODELS = [
    "google/gemini-2.5-flash",
    "z-ai/glm-5",
    "bytedance-seed/seed-2.0-mini"
]

TOPICS = [
    {"agent_type": "adversarial", "topic": "Common misconceptions about async programming in Python", "amount": 20},
    {"agent_type": "qa", "topic": "Designing RESTful APIs that are intuitive and maintainable", "amount": 20},
    {"agent_type": "instruction_following", "topic": "Refactoring poorly written Python code for readability and performance", "amount": 20},
    {"agent_type": "qa", "topic": "Understanding memory management and garbage collection in Python", "amount": 20},
    {"agent_type": "adversarial", "topic": "Common mistakes developers make when implementing authentication and authorization", "amount": 20},
]

async def send(client: httpx.AsyncClient, payload: dict, model: str):
    body = {**payload, "model": model}
    try:
        res = await client.post(f"{SERVER_URL}/dataset/generate", json=body, timeout=600)
        print(f"[{model.split('/')[1][:20]}] [{payload['topic'][:25]}] {res.status_code}")
    except httpx.ReadTimeout:
        print(f"[{model.split('/')[1][:20]}] [{payload['topic'][:25]}] TIMEOUT")
    except httpx.HTTPError as e:
        print(f"[{model.split('/')[1][:20]}] [{payload['topic'][:25]}] ERROR: {e}")

async def main():
    tasks = []
    async with httpx.AsyncClient() as client:
        for model in MODELS:
            for topic in TOPICS:
                tasks.append(send(client, topic, model))
        await asyncio.gather(*tasks)

asyncio.run(main())