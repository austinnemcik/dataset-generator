import itertools
import random

from .types import AgentType


_TECH_LANGUAGE_OPTIONS = [
    "Python",
    "JavaScript",
    "TypeScript",
    "Java",
    "Go",
    "Rust",
    "SQL",
    "Bash",
    "C#",
    "PHP",
]

_TECH_DIFFICULTY_OPTIONS = [
    "beginner",
    "intermediate",
    "advanced",
    "expert",
]

_TECH_BUG_TYPE_OPTIONS = [
    "logic error",
    "type mismatch",
    "async or concurrency bug",
    "state management bug",
    "security flaw",
    "performance regression",
    "data validation bug",
    "API contract mismatch",
    "configuration or dependency issue",
    "resource leak",
    "testing failure",
    "debugging workflow issue",
]

_TECH_SCENARIO_OPTIONS = [
    "bug report",
    "code review comment",
    "failing test case",
    "stack trace investigation",
    "refactor request",
    "incident follow-up",
    "pull request discussion",
    "production hotfix",
]

_GENERIC_SCENARIOS = [
    "everyday situation",
    "advice request",
    "casual question",
    "follow-up clarification",
    "small problem-solving moment",
    "personal reflection",
    "planning conversation",
    "friendly check-in",
]

_GENERIC_INTENTS = [
    "ask for help",
    "share a feeling",
    "make a decision",
    "tell a short story",
    "ask for an opinion",
    "brainstorm ideas",
    "work through uncertainty",
    "continue a previous conversation",
]

_GENERIC_TONES = [
    "warm",
    "casual",
    "supportive",
    "lighthearted",
    "thoughtful",
    "encouraging",
    "curious",
    "calm",
]

_CONVERSATION_PATTERNS = [
    "one direct follow-up question",
    "two-step back and forth",
    "clarification before answering",
    "supportive reply with a follow-up question",
    "casual reply that keeps the conversation going",
    "answer plus a gentle check-in question",
]

_CONVERSATION_TURN_DISTRIBUTIONS = {
    "short": ["2-3 turns", "2-4 turns"],
    "balanced": ["3-4 turns", "4-5 turns", "4-6 turns"],
    "long": ["5-7 turns", "6-8 turns", "7-9 turns"],
    "varied": ["2-3 turns", "2-4 turns", "3-5 turns", "4-6 turns", "6-8 turns"],
}

_INSTRUCTION_FORMATS = [
    "single clear request",
    "multi-step request",
    "short practical task",
    "creative request",
    "everyday planning request",
    "reflective prompt",
]

_TOPIC_PROFILES = [
    {
        "keywords": {"security", "vulnerability", "auth", "injection", "exploit", "deserialization"},
        "languages": ["Python", "JavaScript", "TypeScript", "Java", "Go", "PHP", "SQL", "Bash"],
        "bug_types": [
            "security flaw",
            "data validation bug",
            "API contract mismatch",
            "configuration or dependency issue",
            "logic error",
        ],
        "scenarios": [
            "bug report",
            "incident follow-up",
            "production hotfix",
            "stack trace investigation",
            "code review comment",
        ],
    },
    {
        "keywords": {"review", "debug", "bug", "refactor", "test", "failure", "code"},
        "languages": ["Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "C#", "Bash"],
        "bug_types": [
            "logic error",
            "testing failure",
            "performance regression",
            "async or concurrency bug",
            "state management bug",
            "debugging workflow issue",
            "type mismatch",
        ],
        "scenarios": [
            "code review comment",
            "failing test case",
            "stack trace investigation",
            "pull request discussion",
            "refactor request",
            "production hotfix",
        ],
    },
    {
        "keywords": {"api", "backend", "service", "rest", "endpoint", "integration"},
        "languages": ["Python", "TypeScript", "JavaScript", "Java", "Go", "SQL", "Bash", "C#"],
        "bug_types": [
            "API contract mismatch",
            "data validation bug",
            "configuration or dependency issue",
            "performance regression",
            "logic error",
            "state management bug",
        ],
        "scenarios": [
            "bug report",
            "incident follow-up",
            "pull request discussion",
            "code review comment",
            "production hotfix",
        ],
    },
]

_TECHNICAL_TOPIC_KEYWORDS = {
    "code",
    "debug",
    "bug",
    "software",
    "programming",
    "developer",
    "devops",
    "database",
    "query",
    "api",
    "backend",
    "frontend",
    "stack trace",
    "deployment",
    "compile",
    "runtime",
    "javascript",
    "python",
    "bash",
    "rust",
    "typescript",
    "java",
    "sql",
    "c#",
    "php",
    "go",
}


def load_prompt(name: str) -> str:
    try:
        with open(f"prompts/{name}_agent.txt", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing prompt file: prompts/{name}_agent.txt")


def _topic_profile(topic: str) -> dict | None:
    lowered = topic.casefold()
    best_profile = None
    best_score = 0
    for profile in _TOPIC_PROFILES:
        score = sum(1 for keyword in profile["keywords"] if keyword in lowered)
        if score > best_score:
            best_score = score
            best_profile = profile
    return best_profile


def _merged_options(primary: list[str], fallback: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in primary + fallback:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


def _is_technical_topic(topic: str) -> bool:
    lowered = topic.casefold()
    return any(keyword in lowered for keyword in _TECHNICAL_TOPIC_KEYWORDS)


def _format_blueprint_line(example_index: int, fields: dict[str, str]) -> str:
    formatted_fields = ", ".join(f"{key}={value}" for key, value in fields.items())
    return f"- example {example_index}: {formatted_fields}"


def _build_technical_blueprints(amt: int, *, topic: str, seed: int | None = None) -> list[dict[str, str]]:
    rng = random.Random(seed)
    profile = _topic_profile(topic)
    languages = _merged_options(profile["languages"], _TECH_LANGUAGE_OPTIONS) if profile else list(_TECH_LANGUAGE_OPTIONS)
    bug_types = _merged_options(profile["bug_types"], _TECH_BUG_TYPE_OPTIONS) if profile else list(_TECH_BUG_TYPE_OPTIONS)
    scenarios = _merged_options(profile["scenarios"], _TECH_SCENARIO_OPTIONS) if profile else list(_TECH_SCENARIO_OPTIONS)
    combinations = list(itertools.product(languages, _TECH_DIFFICULTY_OPTIONS, bug_types, scenarios))
    rng.shuffle(combinations)

    blueprints: list[dict[str, str]] = []
    for idx in range(amt):
        language, difficulty, bug_type, scenario = combinations[idx % len(combinations)]
        blueprints.append(
            {
                "example_index": idx + 1,
                "language": language,
                "difficulty": difficulty,
                "bug_type": bug_type,
                "scenario": scenario,
            }
        )
    return blueprints


def _build_conversation_blueprints(
    amt: int,
    *,
    conversation_length_mode: str = "varied",
    seed: int | None = None,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    turn_options = _CONVERSATION_TURN_DISTRIBUTIONS.get(
        conversation_length_mode,
        _CONVERSATION_TURN_DISTRIBUTIONS["varied"],
    )
    combinations = list(
        itertools.product(
            _GENERIC_SCENARIOS,
            _GENERIC_INTENTS,
            _GENERIC_TONES,
            _CONVERSATION_PATTERNS,
            turn_options,
        )
    )
    rng.shuffle(combinations)

    blueprints: list[dict[str, str]] = []
    for idx in range(amt):
        scenario, user_intent, tone, pattern, turn_target = combinations[idx % len(combinations)]
        blueprints.append(
            {
                "example_index": idx + 1,
                "scenario": scenario,
                "user_intent": user_intent,
                "tone": tone,
                "conversation_pattern": pattern,
                "turn_target": turn_target,
            }
        )
    return blueprints


def _build_generic_blueprints(amt: int, *, seed: int | None = None) -> list[dict[str, str]]:
    rng = random.Random(seed)
    combinations = list(itertools.product(_GENERIC_SCENARIOS, _GENERIC_INTENTS, _GENERIC_TONES, _INSTRUCTION_FORMATS))
    rng.shuffle(combinations)

    blueprints: list[dict[str, str]] = []
    for idx in range(amt):
        scenario, user_intent, tone, response_shape = combinations[idx % len(combinations)]
        blueprints.append(
            {
                "example_index": idx + 1,
                "scenario": scenario,
                "user_intent": user_intent,
                "tone": tone,
                "response_shape": response_shape,
            }
        )
    return blueprints


def _build_example_blueprints(
    amt: int,
    *,
    topic: str,
    agent: AgentType,
    conversation_length_mode: str = "varied",
    seed: int | None = None,
) -> list[dict[str, str]]:
    if agent == AgentType.conversation:
        return _build_conversation_blueprints(
            amt,
            conversation_length_mode=conversation_length_mode,
            seed=seed,
        )
    if _is_technical_topic(topic):
        return _build_technical_blueprints(amt, topic=topic, seed=seed)
    return _build_generic_blueprints(amt, seed=seed)


def _build_source_material_block(
    *,
    source_material: str | None,
    source_material_mode: str,
    topic: str,
) -> str:
    if not source_material:
        return ""

    if source_material_mode == "style_only":
        guidance = (
            "Use the source material ONLY for style, tone, voice, pacing, personality, and interaction patterns.\n"
            "Do NOT borrow its subject matter, domain, named entities, facts, settings, or example scenarios unless the topic explicitly asks for them.\n"
            f"The topic ({topic}) still determines what every example is about."
        )
    else:
        guidance = (
            "Use the source material for style, tone, and any relevant facts or details that fit the requested topic.\n"
            "The topic still remains primary. If the source material pulls toward an unrelated domain, keep the topic and only reuse compatible details."
        )

    return f"""
SOURCE MATERIAL:
{source_material}

SOURCE MATERIAL MODE: {source_material_mode}
{guidance}
"""


def build_prompt(
    agent: AgentType,
    topic: str,
    amt: int,
    source_material: str | None = None,
    source_material_mode: str = "content_and_style",
    conversation_length_mode: str = "varied",
    seed: int | None = None,
) -> str:
    blueprint_lines = [
        _format_blueprint_line(
            blueprint["example_index"],
            {key: value for key, value in blueprint.items() if key != "example_index"},
        )
        for blueprint in _build_example_blueprints(
            amt,
            topic=topic,
            agent=agent,
            conversation_length_mode=conversation_length_mode,
            seed=seed,
        )
    ]
    blueprint_block = "\n".join(blueprint_lines)
    source_material_block = _build_source_material_block(
        source_material=source_material,
        source_material_mode=source_material_mode,
        topic=topic,
    )
    conversation_length_block = ""
    if agent == AgentType.conversation:
        conversation_length_block = (
            f"Conversation length mode: {conversation_length_mode}.\n"
            "Keep the conversation history length aligned with each line item's turn_target hint.\n"
            "For varied mode, intentionally mix short, medium, and longer chats instead of making every example the same length.\n"
        )

    base = f"""Generate {amt} diverse examples about {topic}, and make the topic the PRIMARY subject matter of every example.
The topic must determine what the examples are about.
Do not drift into unrelated domains unless the topic or source material explicitly requires it.
Variation hints should change structure, tone, or scenario only. They must not override the topic.
{conversation_length_block}{source_material_block}
Make sure to follow your instructions about formatting and return NOTHING besides JSON.
Returning, or explaining your answers would be considered a FAILURE.
JSON ARRAY ONLY.
Do NOT wrap your output in markdown code fences.
Do NOT include ```json, ``` or any backticks.

Use the structured variation plan below. Produce exactly one example per line item and spread coverage across the requested space so the dataset does not cluster around the same easy pattern.

STRUCTURED VARIATION PLAN:
{blueprint_block}
"""
    if agent == AgentType.domain_specialist:
        return f"""
Generate {amt} training examples grounded in the source material below.
The requested topic is: {topic}
The topic remains the primary subject matter.{_build_source_material_block(source_material=source_material, source_material_mode=source_material_mode, topic=topic)}

Ensure you follow your instructions about formatting your response, and ONLY return JSON.
Do NOT wrap your output in markdown code fences.
Do NOT include ```json, ``` or any backticks.

Use the structured variation plan below. Produce exactly one example per line item and keep the examples grounded in the supplied material.

STRUCTURED VARIATION PLAN:
{blueprint_block}
"""
    return base
