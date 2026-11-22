"""End-to-end tests of decide() with a stand-in model.

These exist because the unit tests passed while SAFE was unreachable: each
component was correct alone and wrong together. They need no API key.
"""

import json

import pytest

from cish import agent, llm
from cish.schema import Verdict


def stub_model(slots: dict, verdict: str):
    def fake(prompt, max_tokens=400):
        if "extract structured facts" in prompt:
            return json.dumps(slots)
        return json.dumps({"verdict": verdict, "rationale": "stub"})
    return fake


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "AUDIT_PATH", tmp_path / "audit.csv")


def run(monkeypatch, slots, model_verdict, requester="e004"):
    monkeypatch.setattr(llm, "complete", stub_model(slots, model_verdict))
    return agent.decide("question", requester)


PERMITTED = dict(data_type="internal_doc", recipient="internal",
                 channel="approved_tool", volume="batch")
CREDENTIAL = dict(data_type="credential", recipient="internal",
                  channel="approved_tool", volume="single_record")


def test_safe_is_reachable(monkeypatch):
    """The regression: SAFE must be returnable for a clearly permitted case."""
    assert run(monkeypatch, PERMITTED, "SAFE").verdict == Verdict.SAFE


def test_model_cannot_override_a_hard_block(monkeypatch):
    assert run(monkeypatch, CREDENTIAL, "SAFE").verdict == Verdict.DO_NOT_SHARE


def test_model_saying_safe_without_a_permissive_clause_is_downgraded(monkeypatch):
    """Customer data to an approved vendor has no permissive clause, so a
    model that says SAFE must be overruled."""
    slots = dict(data_type="customer", recipient="approved_vendor",
                 channel="approved_tool", volume="single_record")
    d = run(monkeypatch, slots, "SAFE", requester="e001")
    assert d.verdict != Verdict.SAFE


def test_unauthorized_requester_is_blocked_even_if_model_says_safe(monkeypatch):
    # e004 is engineering: not entitled to customer data
    slots = dict(data_type="customer", recipient="internal",
                 channel="approved_tool", volume="single_record")
    assert run(monkeypatch, slots, "SAFE", requester="e004").verdict \
        == Verdict.DO_NOT_SHARE


def test_missing_slots_trigger_a_clarifying_question(monkeypatch):
    slots = dict(data_type="customer", recipient="unknown",
                 channel="unknown", volume="unknown")
    d = run(monkeypatch, slots, "SAFE", requester="e001")  # e001 may see customer data
    assert d.clarifying_question
    assert d.verdict == Verdict.CHECK_POLICY


def test_hard_block_wins_even_when_the_channel_is_missing(monkeypatch):
    """Regression for scenario S06: 'export 5,000 customer records to an
    outside agency' never names a channel. The clarifying question used to
    fire first and downgrade a DO NOT SHARE to CHECK POLICY."""
    slots = dict(data_type="customer", recipient="external",
                 channel="unknown", volume="bulk")
    d = run(monkeypatch, slots, "SAFE", requester="e001")
    assert d.verdict == Verdict.DO_NOT_SHARE
    assert d.clarifying_question is None


@pytest.mark.parametrize("volume", ["unknown", "single_record", "batch"])
def test_volume_does_not_change_the_answer_for_a_permitted_case(monkeypatch, volume):
    """Regression for scenario S03: the same Slack question flipped between
    SAFE and CHECK POLICY because the model sometimes said volume=unknown."""
    slots = dict(PERMITTED, volume=volume)
    assert run(monkeypatch, slots, "SAFE").verdict == Verdict.SAFE


def test_missing_channel_still_asks_when_nothing_blocks(monkeypatch):
    slots = dict(data_type="internal_doc", recipient="internal",
                 channel="unknown", volume="unknown")
    d = run(monkeypatch, slots, "SAFE")
    assert d.clarifying_question and d.verdict == Verdict.CHECK_POLICY


def test_skip_model_on_block_saves_the_call_and_keeps_the_verdict(monkeypatch):
    monkeypatch.setenv("CISH_SKIP_MODEL_ON_BLOCK", "1")
    calls = []

    def fake(prompt, max_tokens=400):
        calls.append(prompt)
        if "extract structured facts" in prompt:
            return json.dumps(CREDENTIAL)
        raise AssertionError("verdict call should have been skipped")

    monkeypatch.setattr(llm, "complete", fake)
    d = agent.decide("question", "e004")
    assert d.verdict == Verdict.DO_NOT_SHARE
    assert d.model_verdict is None
    assert len(calls) == 1
