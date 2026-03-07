import pytest


def test_conversation_normalizer_accepts_instruction_ending_with_user_turn():
    pytest.importorskip("sqlmodel")
    from agent.generation import _normalize_generated_example
    from agent.types import AgentType

    normalized = _normalize_generated_example(
        {
            "instruction": "User: Hey, how are you?\nAssistant: I'm good!\nUser: Rough week though.",
            "response": "Aw, I'm sorry. Want to talk through it a little?",
        },
        agent_type=AgentType.conversation,
    )

    assert normalized is not None
    assert normalized["instruction"].endswith("User: Rough week though.")


def test_conversation_normalizer_rejects_instruction_ending_with_assistant_turn():
    pytest.importorskip("sqlmodel")
    from agent.generation import _normalize_generated_example
    from agent.types import AgentType

    normalized = _normalize_generated_example(
        {
            "instruction": "User: Hey\nAssistant: Hi there\nUser: I'm tired\nAssistant: That sounds exhausting",
            "response": "Do you want help planning a gentler evening?",
        },
        agent_type=AgentType.conversation,
    )

    assert normalized is None


def test_conversation_normalizer_strips_assistant_prefix_from_response():
    pytest.importorskip("sqlmodel")
    from agent.generation import _normalize_generated_example
    from agent.types import AgentType

    normalized = _normalize_generated_example(
        {
            "instruction": "User: I had a small win today.\nAssistant: Oh yay.\nUser: I finally finished my homework.",
            "response": "Assistant: That's awesome. You should feel really proud of that.",
        },
        agent_type=AgentType.conversation,
    )

    assert normalized is not None
    assert normalized["response"] == "That's awesome. You should feel really proud of that."
