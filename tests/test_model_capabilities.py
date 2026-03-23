from core.tool_manager import ModelCapabilityDetector


MODELS_DATA = {
    "data": [
        {
            "id": "vision-model",
            "capabilities": {"multimodal": True, "tools": True},
            "info": {"supports_tools": True},
        },
        {
            "id": "text-only",
            "capabilities": {},
            "info": {},
        },
    ]
}


def test_detect_web_search_from_data_true():
    assert ModelCapabilityDetector.detect_web_search_from_data(MODELS_DATA, "vision-model") is True


def test_detect_visual_from_data_true():
    assert ModelCapabilityDetector.detect_visual_from_data(MODELS_DATA, "vision-model") is True


def test_detect_capabilities_from_data_false_for_unknown_model():
    assert ModelCapabilityDetector.detect_web_search_from_data(MODELS_DATA, "unknown") is False
    assert ModelCapabilityDetector.detect_visual_from_data(MODELS_DATA, "unknown") is False
