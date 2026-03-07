import pytest


def test_default_temperature_for_stage_uses_generation_temperature():
    pytest.importorskip("sqlmodel")
    from agent.llm import _default_temperature_for_stage

    assert _default_temperature_for_stage("generation") == 0.70
    assert _default_temperature_for_stage("regeneration_batch") == 0.70


def test_default_temperature_for_stage_uses_lower_naming_and_grading_temperatures():
    pytest.importorskip("sqlmodel")
    from agent.llm import _default_temperature_for_stage

    assert _default_temperature_for_stage("naming") == 0.3
    assert _default_temperature_for_stage("topic_planning") == 0.3
    assert _default_temperature_for_stage("grading_batch") == 0.2
    assert _default_temperature_for_stage("grading_regeneration_batch") == 0.2
