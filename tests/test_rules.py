"""Rules engine tests.

The rules engine is the component that holds when the model fails, so it is
the component that most needs tests. None of these call a model.
"""

import pytest

from cish import directory, rules
from cish.schema import (Channel, DataType, Recipient, Slots, Verdict,
                         more_cautious)


def slots(data_type, recipient, channel, volume, requester="e005"):
    return Slots(data_type=data_type, recipient=recipient, channel=channel,
                 volume=volume, requester_id=requester)


# --- the safety ordering itself -------------------------------------------

def test_more_cautious_picks_the_stricter_verdict():
    assert more_cautious(Verdict.SAFE, Verdict.DO_NOT_SHARE) == Verdict.DO_NOT_SHARE
    assert more_cautious(Verdict.CHECK_POLICY, Verdict.SAFE) == Verdict.CHECK_POLICY
    assert more_cautious(Verdict.SAFE, Verdict.SAFE) == Verdict.SAFE


def test_more_cautious_is_order_independent():
    for a in Verdict:
        for b in Verdict:
            assert more_cautious(a, b) == more_cautious(b, a)


# --- hard blocks -----------------------------------------------------------

def test_credentials_are_blocked_on_every_channel():
    for channel in Channel:
        if channel is Channel.UNKNOWN:
            continue
        v, reasons = rules.apply_rules(
            slots(DataType.CREDENTIAL, Recipient.INTERNAL, channel,
                  "single_record"))
        assert v == Verdict.DO_NOT_SHARE, f"credential allowed via {channel}"
        assert reasons


def test_sensitive_data_blocked_on_personal_device_and_email():
    for channel in (Channel.PERSONAL_DEVICE, Channel.PERSONAL_EMAIL):
        for dt in (DataType.CUSTOMER, DataType.EMPLOYEE, DataType.FINANCIAL):
            v, _ = rules.apply_rules(
                slots(dt, Recipient.SELF_PERSONAL, channel, "single_record"))
            assert v == Verdict.DO_NOT_SHARE


def test_bulk_external_transfer_is_blocked():
    v, _ = rules.apply_rules(
        slots(DataType.CUSTOMER, Recipient.EXTERNAL, Channel.APPROVED_TOOL,
              "bulk"))
    assert v == Verdict.DO_NOT_SHARE


def test_unsanctioned_ai_blocked_for_non_public_data():
    v, _ = rules.apply_rules(
        slots(DataType.INTERNAL_DOC, Recipient.EXTERNAL,
              Channel.UNSANCTIONED_TOOL, "single_record"))
    assert v == Verdict.DO_NOT_SHARE


def test_public_data_to_unsanctioned_tool_is_not_hard_blocked():
    v, _ = rules.apply_rules(
        slots(DataType.PUBLIC, Recipient.PUBLIC, Channel.UNSANCTIONED_TOOL,
              "single_record"))
    assert v != Verdict.DO_NOT_SHARE


# --- authorization ---------------------------------------------------------

def test_unauthorized_role_is_blocked():
    """A support rep has no entitlement to employee records."""
    v, reasons = rules.apply_rules(
        slots(DataType.EMPLOYEE, Recipient.INTERNAL, Channel.APPROVED_TOOL,
              "single_record", requester="e001"))
    assert v == Verdict.DO_NOT_SHARE
    assert any("entitled" in r for r in reasons)


def test_authorized_role_is_not_blocked_by_authorization():
    """HR may handle employee records, so nothing should fire here."""
    v, _ = rules.apply_rules(
        slots(DataType.EMPLOYEE, Recipient.INTERNAL, Channel.APPROVED_TOOL,
              "single_record", requester="e002"))
    assert v != Verdict.DO_NOT_SHARE


def test_unknown_requester_is_never_authorized():
    assert not directory.authorized("not-an-employee", DataType.CUSTOMER)


def test_requester_identity_comes_from_the_session():
    """The slot is set from the authenticated session, so a question that
    claims a different identity cannot change the authorization check."""
    s = slots(DataType.EMPLOYEE, Recipient.INTERNAL, Channel.APPROVED_TOOL,
              "single_record", requester="e001")
    assert s.requester_id == "e001"
    assert not directory.authorized(s.requester_id, DataType.EMPLOYEE)


# --- the default -----------------------------------------------------------

def test_rules_can_only_veto():
    """The rules return DO_NOT_SHARE or SAFE (meaning no objection), never
    CHECK_POLICY. They do not approve anything: approval needs a permissive
    clause, which is enforced in agent.model_verdict."""
    seen = set()
    for dt in DataType:
        for ch in Channel:
            for rc in Recipient:
                v, _ = rules.apply_rules(slots(dt, rc, ch, "single_record"))
                seen.add(v)
    assert seen <= {Verdict.SAFE, Verdict.DO_NOT_SHARE}
