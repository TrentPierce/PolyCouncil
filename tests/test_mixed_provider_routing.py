import asyncio
import json

import council


def test_vote_one_routes_to_voter_provider_and_model(monkeypatch):
    calls = []

    async def fake_call_model(session, provider, model, user_prompt, **kwargs):
        calls.append((provider.provider_type, provider.base_url, model))
        ballot = {
            "scores": {
                "1": {"correctness": 5, "relevance": 3, "specificity": 3, "safety": 2, "conciseness": 1}
            },
            "final_pick": 1,
            "reasoning": "ok"
        }
        return json.dumps(ballot)

    monkeypatch.setattr(council, "call_model", fake_call_model)

    voter_provider = council.make_provider_config(council.PROVIDER_LM_STUDIO, "http://localhost:1234", "", "")
    peer_provider = council.make_provider_config(council.PROVIDER_OPENAI_COMPAT, "https://openrouter.ai/api/v1", "k", "models", api_service=council.API_SERVICE_OPENROUTER)

    voter_entry = {"id": "LM Studio :: local-model", "model": "local-model", "provider": voter_provider}
    model_entries = [
        voter_entry,
        {"id": "OpenRouter :: gpt", "model": "openrouter/gpt", "provider": peer_provider},
    ]
    answers = {
        "LM Studio :: local-model": "local answer",
        "OpenRouter :: gpt": "remote answer",
    }

    voter_id, ballot, msg = asyncio.run(
        council.vote_one(None, voter_entry, "q", answers, model_entries)
    )

    assert voter_id == "LM Studio :: local-model"
    assert ballot is not None
    assert ballot["scores"]["OpenRouter :: gpt"]["correctness"] == 5
    assert calls[0][2] == "local-model"
    assert calls[0][0] == council.PROVIDER_LM_STUDIO
    assert "Valid ballot" in msg
