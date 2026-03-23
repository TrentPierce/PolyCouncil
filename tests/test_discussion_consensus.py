from core.discussion_manager import DiscussionManager, PROVIDER_OPENAI_COMPAT


def _manager():
    return DiscussionManager(
        provider_type=PROVIDER_OPENAI_COMPAT,
        base_url="http://example.com",
        api_key="secret",
        model_path="v1/models",
        agents=[
            {"name": "A", "model": "m1", "is_active": True},
            {"name": "B", "model": "m2", "is_active": True},
            {"name": "C", "model": "m3", "is_active": True},
        ],
        user_prompt="hello",
    )


def test_consensus_requires_strong_multi_agent_signal():
    mgr = _manager()
    mgr.turn_count = 3
    history = [
        {"agent": "A", "message": "I agree that we should explore tradeoffs."},
        {"agent": "B", "message": "However, I recommend another path."},
        {"agent": "C", "message": "Summary: both options have value."},
        {"agent": "A", "message": "I agree we need more evidence."},
        {"agent": "B", "message": "Still disagree for now."},
        {"agent": "C", "message": "Let's continue."},
    ]
    assert mgr._check_consensus(history) is False


def test_consensus_accepts_multiple_strong_alignment_messages():
    mgr = _manager()
    mgr.turn_count = 4
    history = [
        {"agent": "A", "message": "We agree the safer rollout is staged."},
        {"agent": "B", "message": "Consensus reached: staged rollout with monitoring."},
        {"agent": "C", "message": "Our conclusion is the same staged approach."},
        {"agent": "A", "message": "Shared conclusion: stage the deployment."},
        {"agent": "B", "message": "Joint recommendation: release in phases."},
        {"agent": "C", "message": "Aligned on phased rollout and checks."},
    ]
    assert mgr._check_consensus(history) is True
