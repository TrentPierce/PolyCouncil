from council import normalize_rubric_weights, safe_load_vote_json, validate_ballot


def test_vote_json_parsing_with_wrapped_text():
    parsed = safe_load_vote_json('prefix {"scores": {"1": {"correctness": 5, "relevance": 3, "specificity": 3, "safety": 2, "conciseness": 1}}, "final_pick": 1} suffix')
    assert parsed is not None
    ok, _ = validate_ballot({1: "m1"}, parsed)
    assert ok is True


def test_normalize_rubric_weights_ignores_bad_values():
    weights = normalize_rubric_weights({"correctness": "7", "safety": -2, "unknown": 99})
    assert weights["correctness"] == 7
    assert weights["safety"] == 0
    assert "unknown" not in weights
