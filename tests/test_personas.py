from pathlib import Path

from core import personas


def test_merge_persona_library_prefers_builtin_and_sorts(tmp_path: Path):
    default_path = tmp_path / "default.json"
    user_path = tmp_path / "user.json"
    default_path.write_text(
        '[{"id":"d1","name":"Built In","prompt_instruction":"Default prompt"}]',
        encoding="utf-8",
    )
    user_path.write_text(
        '[{"id":"u1","name":"Custom","prompt_instruction":"User prompt"}]',
        encoding="utf-8",
    )

    merged = personas.merge_persona_library(
        [{"name": "Built In", "prompt": "Legacy override", "builtin": False}],
        default_path=default_path,
        user_path=user_path,
        default_personas=[{"name": "None", "prompt": None, "builtin": True}],
    )

    assert [entry["name"] for entry in merged] == ["None", "Built In", "Custom"]
    assert personas.persona_prompt(merged, "Built In") == "Default prompt"
    assert personas.persona_prompt(merged, "Custom") == "User prompt"


def test_cleanup_and_rename_persona_assignments():
    library = [
        {"name": "None", "prompt": None, "builtin": True},
        {"name": "Analyst", "prompt": "Analyze", "builtin": False},
    ]
    assignments = {"model-a": "Analyst", "model-b": "Missing"}

    cleaned, dirty = personas.cleanup_persona_assignments(library, assignments)

    assert dirty is True
    assert cleaned == {"model-a": "Analyst", "model-b": "None"}
    assert personas.rename_persona_assignments(cleaned, "Analyst", "Reviewer")["model-a"] == "Reviewer"
    assert personas.clear_persona_assignment(cleaned, "Analyst")["model-a"] == "None"


def test_build_persona_config_prefers_file_ids_and_falls_back_to_prompt(tmp_path: Path):
    default_path = tmp_path / "default.json"
    user_path = tmp_path / "user.json"
    default_path.write_text(
        '[{"id":"d1","name":"Built In","prompt_instruction":"Default prompt"}]',
        encoding="utf-8",
    )
    user_path.write_text(
        '[{"id":"u1","name":"Custom","prompt_instruction":"User prompt"}]',
        encoding="utf-8",
    )

    personas_list = [
        {"name": "None", "prompt": None, "builtin": True},
        {"name": "Ephemeral", "prompt": "One-shot prompt", "builtin": False},
    ]

    built_in = personas.build_persona_config(
        "Built In",
        personas_list,
        default_path=default_path,
        user_path=user_path,
    )
    custom = personas.build_persona_config(
        "Custom",
        personas_list,
        default_path=default_path,
        user_path=user_path,
    )
    ephemeral = personas.build_persona_config(
        "Ephemeral",
        personas_list,
        default_path=default_path,
        user_path=user_path,
    )

    assert built_in == {"source": "default", "id": "d1", "one_time_prompt": ""}
    assert custom == {"source": "user_custom", "id": "u1", "one_time_prompt": ""}
    assert ephemeral == {"source": "one_time", "id": None, "one_time_prompt": "One-shot prompt"}
