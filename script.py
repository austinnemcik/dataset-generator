import asyncio
import httpx

SERVER_URL = "http://localhost:8000"

payloads = [
    {"agent_type": "qa", "topic": "Common Python debugging techniques and error handling", "amount": 20},
    {"agent_type": "qa", "topic": "Tradeoffs between relational and NoSQL databases", "amount": 20},
    {"agent_type": "qa", "topic": "How neural networks learn through backpropagation", "amount": 20},
    {"agent_type": "instruction_following", "topic": "REST API design best practices", "amount": 20},
    {"agent_type": "qa", "topic": "Git workflow and version control strategies", "amount": 20},
    {"agent_type": "qa", "topic": "Object oriented programming principles in Python", "amount": 20},
    {"agent_type": "qa", "topic": "Time and space complexity in algorithm analysis", "amount": 20},
    {"agent_type": "qa", "topic": "Docker containerization concepts and usage", "amount": 20},
    {"agent_type": "adversarial", "topic": "Common security vulnerabilities in web applications", "amount": 20},
    {"agent_type": "instruction_following", "topic": "Linux command line fundamentals", "amount": 20},
]

async def send(client, payload):
    res = await client.post(f"{SERVER_URL}/dataset/generate", json=payload, timeout=300)
    print(f"[{payload['topic'][:30]}] {res.status_code}")

async def main():
    async with httpx.AsyncClient() as client:
        tasks = []
        for payload in payloads:
            tasks.append(send(client, payload))
            await asyncio.sleep(0.5)  # stagger by half a second
        await asyncio.gather(*tasks)

asyncio.run(main())