import pytest


def test_build_prompt_for_casual_conversation_topic_avoids_technical_blueprints():
    pytest.importorskip("sqlmodel")
    from agent.prompts import build_prompt
    from agent.types import AgentType

    prompt = build_prompt(
        AgentType.conversation,
        "Warm casual conversation with follow-up questions",
        3,
        seed=1,
    )

    assert "conversation_pattern=" in prompt
    assert "turn_target=" in prompt
    assert "language=" not in prompt
    assert "bug_type=" not in prompt
    assert "stack trace" not in prompt.casefold()
    assert "PRIMARY subject matter" in prompt


def test_build_prompt_for_generic_nontechnical_topic_uses_neutral_variation_axes():
    pytest.importorskip("sqlmodel")
    from agent.prompts import build_prompt
    from agent.types import AgentType

    prompt = build_prompt(
        AgentType.instruction_following,
        "Friendly everyday Q&A and advice",
        2,
        seed=1,
    )

    assert "response_shape=" in prompt
    assert "language=" not in prompt
    assert "bug_type=" not in prompt


def test_build_prompt_for_technical_topic_keeps_technical_blueprints():
    pytest.importorskip("sqlmodel")
    from agent.prompts import build_prompt
    from agent.types import AgentType

    prompt = build_prompt(
        AgentType.qa,
        "Code review and debugging",
        2,
        seed=1,
    )

    assert "language=" in prompt
    assert "bug_type=" in prompt


def test_build_prompt_style_only_uses_source_material_for_voice_not_subject():
    pytest.importorskip("sqlmodel")
    from agent.prompts import build_prompt
    from agent.types import AgentType

    prompt = build_prompt(
        AgentType.conversation,
        "Warm casual conversation with follow-up questions",
        2,
        source_material="A terse cybersecurity incident report about API key leakage.",
        source_material_mode="style_only",
        seed=1,
    )

    assert "SOURCE MATERIAL MODE: style_only" in prompt
    assert "Use the source material ONLY for style, tone, voice, pacing, personality, and interaction patterns." in prompt
    assert "Do NOT borrow its subject matter" in prompt


def test_build_prompt_content_and_style_allows_compatible_grounding():
    pytest.importorskip("sqlmodel")
    from agent.prompts import build_prompt
    from agent.types import AgentType

    prompt = build_prompt(
        AgentType.instruction_following,
        "Friendly everyday Q&A and advice",
        2,
        source_material="Helpful examples of warm, encouraging conversational replies.",
        source_material_mode="content_and_style",
        seed=1,
    )

    assert "SOURCE MATERIAL MODE: content_and_style" in prompt
    assert "Use the source material for style, tone, and any relevant facts or details that fit the requested topic." in prompt


def test_build_prompt_for_conversation_varied_length_mentions_mixed_turn_targets():
    pytest.importorskip("sqlmodel")
    from agent.prompts import build_prompt
    from agent.types import AgentType

    prompt = build_prompt(
        AgentType.conversation,
        "Warm casual conversation with follow-up questions",
        5,
        conversation_length_mode="varied",
        seed=1,
    )

    assert "Conversation length mode: varied." in prompt
    assert "turn_target=" in prompt


def test_build_prompt_for_conversation_long_length_uses_longer_turn_targets():
    pytest.importorskip("sqlmodel")
    from agent.prompts import build_prompt
    from agent.types import AgentType

    prompt = build_prompt(
        AgentType.conversation,
        "Warm casual conversation with follow-up questions",
        4,
        conversation_length_mode="long",
        seed=1,
    )

    assert "Conversation length mode: long." in prompt
    assert "turn_target=6-8 turns" in prompt or "turn_target=7-9 turns" in prompt or "turn_target=5-7 turns" in prompt


def test_build_prompt_for_technical_conversation_still_uses_conversation_turn_targets():
    pytest.importorskip("sqlmodel")
    from agent.prompts import build_prompt
    from agent.types import AgentType

    prompt = build_prompt(
        AgentType.conversation,
        "Debugging a flaky API integration with follow-up questions",
        3,
        conversation_length_mode="balanced",
        seed=1,
    )

    assert "turn_target=" in prompt
    assert "conversation_pattern=" in prompt
