"""The harness must not score failed calls as wrong answers."""

import evaluate
from cish import agent
from cish.schema import Decision, Verdict


def fake_decide(outcomes):
    it = iter(outcomes)

    def decide(question, requester):
        o = next(it)
        if isinstance(o, Exception):
            raise o
        return Decision(verdict=o, rationale="x")
    return decide


SC = {"id": "T1", "requester": "e001", "expected": "DO NOT SHARE",
      "question": "q", "paraphrases": ["a", "b"]}


def test_errors_are_not_counted_as_wrong_answers(monkeypatch):
    monkeypatch.setattr(agent, "decide", fake_decide(
        [Verdict.DO_NOT_SHARE, RuntimeError("429"), Verdict.DO_NOT_SHARE]))
    s = evaluate.evaluate([SC])
    r = s["results"][0]
    assert r["correct"] and r["scored"]
    assert not r["consistent"]          # a failed call means not fully consistent
    assert s["error_runs"] == 1 and s["error_rate"] > 0
    assert s["accuracy"] == 1.0


def test_all_errors_is_unscored_not_inaccurate(monkeypatch):
    monkeypatch.setattr(agent, "decide", fake_decide([RuntimeError("x")] * 3))
    s = evaluate.evaluate([SC])
    assert not s["results"][0]["scored"]
    assert s["error_rate"] == 1.0


def test_unsafe_safe_detected(monkeypatch):
    monkeypatch.setattr(agent, "decide", fake_decide(
        [Verdict.DO_NOT_SHARE, Verdict.SAFE, Verdict.DO_NOT_SHARE]))
    assert evaluate.evaluate([SC])["unsafe_safe_rate"] == 1.0


def test_error_rate_is_a_release_criterion():
    assert "error_rate" in evaluate.RELEASE_CRITERIA
