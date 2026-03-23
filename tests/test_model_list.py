from gui.model_list import build_model_badge_text, make_unique_display_model_name, persona_button_state


def test_make_unique_display_model_name_increments_suffix():
    existing = {
        "LM Studio :: llama",
        "LM Studio :: llama (2)",
    }

    assert make_unique_display_model_name("llama", "LM Studio", existing) == "LM Studio :: llama (3)"


def test_build_model_badge_text_includes_capabilities_and_latency():
    badge = build_model_badge_text(
        "OpenAI",
        capabilities={"visual": True, "web_search": True},
        latency_ms=182.9,
    )

    assert badge == "OpenAI | vision, web | 182 ms"


def test_persona_button_state_truncates_long_names():
    text, tooltip = persona_button_state("Very long persona name")

    assert text == "Very long .."
    assert tooltip == "Very long persona name"
