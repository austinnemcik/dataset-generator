import json
import requests
from agents import Agent, Runner
from dotenv import load_dotenv
import os
from enum import Enum
from openai import OpenAI
import numpy as np
load_dotenv()
_SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
THRESHOLD = 0.8
client = OpenAI()

def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def load_prompt(name: str) -> str:
    with open(f"prompts/{name}.txt", "r", encoding="utf-8") as file:
        return file.read()
    
def cosine_similarity(a: list[float], b: list[float]):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_duplicate(new_embedding: list[float], existing_embeddings: list[list[float]], threshold: float = THRESHOLD):
    for existing in existing_embeddings: 
        if cosine_similarity(new_embedding, existing) >= threshold:
            return True
    return False

naming_agent = Agent(name="Naming Agent", instructions=load_prompt("naming_agent"))
qa_agent = Agent(name="Q&A Agent", instructions=load_prompt("qa_agent"))
instruction_following_agent = Agent(name="Instruction Following Agent", instructions=load_prompt("instruction_following_agent"))
domain_specialist_agent = Agent(name="Domain Specialist Agent", instructions=load_prompt("domain_specialist_agent"))
style_agent = Agent(name="Style Agent", instructions=load_prompt("style_agent"))
adversarial_agent = Agent(name="Adversarial Agent", instructions=load_prompt("adversarial_agent"))
conversation_agent = Agent(name="Conversation Agent", instructions=load_prompt("conversation_agent"))

class AgentType(str, Enum):
    qa = "qa"
    instruction_following = "instruction_following"
    domain_specialist = "domain_specialist"
    style = "style"
    adversarial = "adversarial"
    conversation = "conversation"

agent_map = {
    AgentType.qa: qa_agent,
    AgentType.instruction_following: instruction_following_agent,
    AgentType.domain_specialist: domain_specialist_agent,
    AgentType.style: style_agent,
    AgentType.adversarial: adversarial_agent,
    AgentType.conversation: conversation_agent
}

def save_responses(examples: list[dict]):
    # Ask naming_agent for name + description
    naming_prompt = f"""
Generate a dataset name and description for the following examples.
Return STRICT JSON: {{"name": "...", "description": "..."}}

Examples JSON:
{json.dumps(examples)}
"""
    naming_result = Runner.run_sync(naming_agent, naming_prompt)

    try:
        meta = json.loads(naming_result.final_output)
    except json.JSONDecodeError as e:
        print(f"[save_responses] Failed to parse naming_agent output: {e}")
        print("naming_agent output was:", naming_result.final_output)
        return

    # Build ingest payload
    payload = {
        "dataset_name": meta["name"],
        "dataset_description": meta["description"],
        "dataset_id": 0,
        "example": examples,
    }

    # POST to /ingest
    try:
        res = requests.post(f"{_SERVER_URL}/dataset/ingest", json=payload, timeout=30)
        res.raise_for_status()
        result = res.json()
        print(f"[save_responses] Ingest response: {result}")
        return result
    except requests.RequestException as e:
        print(f"[save_responses] Failed to POST to /ingest: {e}")

def generate_dataset(agent_type: AgentType, topic: str, amt: int, source_material: str | None = None) -> list[dict]:
    agent = agent_map[agent_type]
    prompt = build_prompt(agent, topic, amt, source_material)
    result = Runner.run_sync(agent, prompt)
    
    try: 
        examples = json.loads(result.final_output)
    except json.JSONDecodeError:
        raise ValueError("Agent returned malformed JSON")
    
    valid = [
        ex for ex in examples 
        if isinstance(ex, dict)
        and "instruction" in ex
        and "response" in ex
        and len(ex["instruction"]) > 2
        and len(ex["response"]) > 2
    ]
    return valid

def build_prompt(agent: AgentType, topic: str, amt: int, source_material: str | None = None) -> str:
    base = f"Generate {amt} diverse examples about {topic}, make sure to follow your instructions about formatting and return NOTHING besides JSON. Returning, or explaining your answers would be considered a FAILURE. JSON ARRAY ONLY"
    if agent == AgentType.domain_specialist:
        return f"""
Using ONLY the following source material, generate {amt} training examples. 

SOURCE MATERIAL:
{source_material}

Ensure you follow your instructions about formatting your response,  and ONLY return only JSON
"""
    return base
