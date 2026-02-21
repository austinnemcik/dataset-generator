import json
import requests
from agents import Agent, Runner
from dotenv import load_dotenv

load_dotenv()

def load_prompt(name: str) -> str:
    with open(f"prompts/{name}.txt", "r", encoding="utf-8") as file:
        return file.read()

naming_agent = Agent(name="Naming Agent", instructions=load_prompt("naming_agent"))

qa_agent = Agent(name="Q&A Agent", instructions=load_prompt("qa_agent"))
instruction_following_agent = Agent(name="Instruction Following Agent", instructions=load_prompt("instruction_following_agent"))
domain_specialist_agent = Agent(name="Domain Specialist Agent", instructions=load_prompt("domain_specialist_agent"))
style_agent = Agent(name="Style Agent", instructions=load_prompt("style_agent"))
adversarial_agent = Agent(name="Adversarial Agent", instructions=load_prompt("adversarial_agent"))
conversation_agent = Agent(name="Conversation Agent", instructions=load_prompt("conversation_agent"))

GENERATOR_AGENTS = [
    (qa_agent,                   "Generate 10 Q&A pairs about Python basics."),
    (instruction_following_agent,"Generate 10 instruction-following examples covering a variety of tasks and output formats."),
    (domain_specialist_agent,    "Generate 10 expert-level examples about machine learning concepts."),
    (style_agent,                "Generate 10 writing style transformation examples across different tones and formats."),
    (adversarial_agent,          "Generate 10 adversarial examples with tricky, ambiguous, or edge-case instructions."),
    (conversation_agent,         "Generate 10 multi-turn conversation examples on everyday topics."),
]

def save_responses(response_text: str):
    # 1) Parse the agent output (examples)
    try:
        examples = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"[save_responses] Failed to parse agent output as JSON: {e}")
        return

    # 2) Ask naming_agent for name + description
    naming_prompt = f"""
Generate a dataset name and description for the following examples.
Return STRICT JSON: {{"name": "...", "description": "..."}}

Examples JSON:
{response_text}
"""
    naming_result = Runner.run_sync(naming_agent, naming_prompt)

    try:
        meta = json.loads(naming_result.final_output)
    except json.JSONDecodeError as e:
        print(f"[save_responses] Failed to parse naming_agent output: {e}")
        print("naming_agent output was:", naming_result.final_output)
        return

    # 3) Build ingest payload
    payload = {
        "dataset_name": meta["name"],
        "dataset_description": meta["description"],
        "dataset_id": 0,
        "example": examples,
    }

    # 4) POST to /ingest
    try:
        res = requests.post("http://localhost:8000/ingest", json=payload, timeout=30)
        res.raise_for_status()
        result = res.json()
        print(f"[save_responses] Ingest response: {result}")
        return result
    except requests.RequestException as e:
        print(f"[save_responses] Failed to POST to /ingest: {e}")

for agent, prompt in GENERATOR_AGENTS:
    print(f"[runner] Running {agent.name}...")
    result = Runner.run_sync(agent, prompt)
    save_responses(result.final_output)