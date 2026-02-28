import itertools
import random

from .types import AgentType


_LANGUAGE_OPTIONS = [
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

_DIFFICULTY_OPTIONS = [
    "beginner",
    "intermediate",
    "advanced",
    "expert",
]

_BUG_TYPE_OPTIONS = [
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

_SCENARIO_OPTIONS = [
    "bug report",
    "code review comment",
    "failing test case",
    "stack trace investigation",
    "refactor request",
    "incident follow-up",
    "pull request discussion",
    "production hotfix",
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
            "integration failure",
            "incident follow-up",
            "pull request discussion",
            "code review comment",
        ],
    },
]


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


def _build_example_blueprints(
    amt: int, *, topic: str, seed: int | None = None
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    profile = _topic_profile(topic)
    languages = (
        _merged_options(profile["languages"], _LANGUAGE_OPTIONS)
        if profile
        else list(_LANGUAGE_OPTIONS)
    )
    bug_types = (
        _merged_options(profile["bug_types"], _BUG_TYPE_OPTIONS)
        if profile
        else list(_BUG_TYPE_OPTIONS)
    )
    scenarios = (
        _merged_options(profile["scenarios"], _SCENARIO_OPTIONS)
        if profile
        else list(_SCENARIO_OPTIONS)
    )
    combinations = list(
        itertools.product(
            languages,
            _DIFFICULTY_OPTIONS,
            bug_types,
            scenarios,
        )
    )
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


def build_prompt(
    agent: AgentType,
    topic: str,
    amt: int,
    source_material: str | None = None,
    seed: int | None = None,
) -> str:
    blueprint_lines = []
    for blueprint in _build_example_blueprints(amt, topic=topic, seed=seed):
        blueprint_lines.append(
            f'- example {blueprint["example_index"]}: '
            f'language={blueprint["language"]}, '
            f'difficulty={blueprint["difficulty"]}, '
            f'bug_type={blueprint["bug_type"]}, '
            f'scenario={blueprint["scenario"]}'
        )
    blueprint_block = "\n".join(blueprint_lines)

    base = f"""Generate {amt} diverse examples about {topic}, make sure to follow your instructions about formatting and return NOTHING besides JSON.
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
Using ONLY the following source material, generate {amt} training examples.

SOURCE MATERIAL:
{source_material}

Ensure you follow your instructions about formatting your response, and ONLY return JSON.
Do NOT wrap your output in markdown code fences.
Do NOT include ```json, ``` or any backticks.

Use the structured variation plan below. Produce exactly one example per line item and keep the examples grounded in the supplied material.

STRUCTURED VARIATION PLAN:
{blueprint_block}
"""
    return base

