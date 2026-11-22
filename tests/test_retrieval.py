"""Retrieval tests.

These pin the behaviour that matters: the clause a human would cite for a
case should be in the top results, and a permissive clause should only come
back when the action really is permitted.

Runs against the TF-IDF backend, which needs no model download.
"""

import pytest

from cish import retrieval
from cish.index import contains_permissive_clause, retrieve_policy
from cish.schema import Channel, DataType, Recipient, Slots, Volume


@pytest.fixture(autouse=True)
def tfidf_backend(monkeypatch):
    monkeypatch.setenv("CISH_RETRIEVER", "tfidf")
    retrieval.reset_cache()
    yield
    retrieval.reset_cache()


def q(data_type, recipient, channel, volume):
    return Slots(data_type=data_type, recipient=recipient, channel=channel,
                 volume=volume, requester_id="e005")


# (slots, clause that should surface in the top results)
GOVERNING_CASES = [
    (q(DataType.CREDENTIAL, Recipient.INTERNAL, Channel.APPROVED_TOOL,
       Volume.SINGLE_RECORD), "AUP-4.1"),
    (q(DataType.CUSTOMER, Recipient.EXTERNAL, Channel.UNSANCTIONED_TOOL,
       Volume.BULK), "AI-3.1"),
    (q(DataType.FINANCIAL, Recipient.SELF_PERSONAL, Channel.PERSONAL_DEVICE,
       Volume.SINGLE_RECORD), "AUP-3.1"),
    (q(DataType.EMPLOYEE, Recipient.SELF_PERSONAL, Channel.PERSONAL_EMAIL,
       Volume.SINGLE_RECORD), "AUP-3.1"),
    (q(DataType.CUSTOMER, Recipient.SELF_PERSONAL, Channel.REMOVABLE_MEDIA,
       Volume.BATCH), "AUP-3.2"),
]


@pytest.mark.parametrize("slots,expected_clause", GOVERNING_CASES)
def test_governing_clause_is_retrieved(slots, expected_clause):
    provisions = retrieve_policy(slots)
    ids = [p.clause_id for p in provisions]
    assert expected_clause in ids, \
        f"expected {expected_clause} in {ids} for {slots.as_retrieval_query()}"


def test_retrieval_returns_at_most_k():
    provisions = retrieve_policy(
        q(DataType.CUSTOMER, Recipient.INTERNAL, Channel.APPROVED_TOOL,
          Volume.BATCH), k=3)
    assert len(provisions) <= 3


def test_results_are_sorted_by_score():
    provisions = retrieve_policy(
        q(DataType.CUSTOMER, Recipient.INTERNAL, Channel.APPROVED_TOOL,
          Volume.BATCH))
    scores = [p.score for p in provisions]
    assert scores == sorted(scores, reverse=True)


def test_permissive_clause_found_for_a_permitted_action():
    """Internal documents through an approved tool is the one clearly
    permitted path in this corpus, so SAFE must be reachable there."""
    provisions = retrieve_policy(
        q(DataType.INTERNAL_DOC, Recipient.INTERNAL, Channel.APPROVED_TOOL,
          Volume.BATCH))
    assert contains_permissive_clause(provisions)


def test_no_permissive_clause_for_a_prohibited_action():
    """If a permissive clause surfaced here, agent.decide() could return
    SAFE for sending customer data to a personal phone."""
    provisions = retrieve_policy(
        q(DataType.CUSTOMER, Recipient.SELF_PERSONAL, Channel.PERSONAL_DEVICE,
          Volume.BULK))
    assert not contains_permissive_clause(provisions)


def test_unknown_slots_do_not_crash_retrieval():
    provisions = retrieve_policy(
        q(DataType.UNKNOWN, Recipient.UNKNOWN, Channel.UNKNOWN,
          Volume.UNKNOWN))
    assert isinstance(provisions, list)


def test_every_provision_carries_a_citation():
    """A recommendation with no clause id cannot be checked by the employee,
    which defeats the point of grounding."""
    provisions = retrieve_policy(
        q(DataType.CUSTOMER, Recipient.EXTERNAL, Channel.UNSANCTIONED_TOOL,
          Volume.BULK))
    assert provisions
    for p in provisions:
        assert p.clause_id and p.doc_title and p.text
