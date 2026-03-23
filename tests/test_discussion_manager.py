import asyncio

from core.discussion_manager import DiscussionManager, PROVIDER_OPENAI_COMPAT


def _manager(**kwargs):
    return DiscussionManager(
        provider_type=kwargs.get("provider_type", PROVIDER_OPENAI_COMPAT),
        base_url=kwargs.get("base_url", "http://example.com"),
        api_key=kwargs.get("api_key", "secret"),
        model_path=kwargs.get("model_path", "v1/models"),
        agents=kwargs.get("agents", []),
        user_prompt=kwargs.get("user_prompt", "hello"),
        is_cancelled=kwargs.get("is_cancelled"),
    )


def test_headers_include_api_key_for_openai_compatible():
    mgr = _manager()
    headers = mgr._headers()
    assert headers["Authorization"] == "Bearer secret"


def test_chat_url_for_ollama():
    mgr = _manager(provider_type="ollama", base_url="http://localhost:11434")
    assert mgr._chat_url() == "http://localhost:11434/api/chat"


def test_run_discussion_respects_cancellation_without_agents():
    mgr = _manager(is_cancelled=lambda: True)
    transcript, synthesis = asyncio.run(mgr.run_discussion())
    assert transcript == []
    assert synthesis is None
