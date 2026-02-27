import asyncio
import httpx

SERVER_URL = "http://localhost:8000"

MODELS = [
    "openai/gpt-5-mini",
    "openai/gpt-4.1-mini",
    "deepseek/deepseek-v3.2",
    "google/gemini-2.5-flash",
    "z-ai/glm-5",
    "bytedance-seed/seed-2.0-mini",
]

TOPICS = [
    {"agent_type": "qa", "topic": "Common Python debugging techniques and error handling", "amount": 20},
    {"agent_type": "qa", "topic": "Tradeoffs between relational and NoSQL databases", "amount": 20},
    {"agent_type": "qa", "topic": "Object oriented programming principles in Python", "amount": 20},
    {"agent_type": "qa", "topic": "Git workflow and version control strategies", "amount": 20},
    {"agent_type": "adversarial", "topic": "Common security vulnerabilities in web applications", "amount": 20},
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