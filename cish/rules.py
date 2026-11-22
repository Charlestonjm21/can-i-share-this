"""Deterministic rule layer.

These rules do not consult the language model. They are the component that
makes the conservative default in decide() trustworthy: if the model is wrong,
the rules still hold. They can only veto, never approve. Implements HARD_BLOCK_RULES from the Part 2 design.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from . import directory
from .schema import Channel, DataType, Recipient, Slots, Verdict

SENSITIVE = {DataType.CUSTOMER, DataType.EMPLOYEE, DataType.FINANCIAL}


def _credential_anywhere(s: Slots) -> Optional[str]:
    if s.data_type == DataType.CREDENTIAL:
        return ("Credentials may not be shared through any channel "
                "(AUP-4.1).")
    return None


def _personal_device_or_email(s: Slots) -> Optional[str]:
    if s.channel in (Channel.PERSONAL_DEVICE, Channel.PERSONAL_EMAIL) \
            and s.data_type in SENSITIVE:
        return ("Confidential information may not be moved to a personal "
                "device or personal email account (AUP-3.1).")
    return None


def _removable_media(s: Slots) -> Optional[str]:
    if s.channel == Channel.REMOVABLE_MEDIA and s.data_type in SENSITIVE:
        return ("Copying confidential information to removable media "
                "requires written Information Security approval (AUP-3.2).")
    return None


def _bulk_external(s: Slots) -> Optional[str]:
    if s.recipient == Recipient.EXTERNAL and s.volume.value == "bulk" \
            and s.data_type in SENSITIVE:
        return ("Bulk transfers of confidential information to external "
                "recipients require written approval (AUP-5.2).")
    return None


def _unsanctioned_ai(s: Slots) -> Optional[str]:
    if s.channel == Channel.UNSANCTIONED_TOOL and s.data_type != DataType.PUBLIC:
        return ("Company information may not be submitted to consumer or "
                "unapproved AI services (AI-3.1).")
    return None


def _not_authorized(s: Slots) -> Optional[str]:
    if s.data_type in (DataType.PUBLIC, DataType.UNKNOWN):
        return None
    if not directory.authorized(s.requester_id, s.data_type):
        return (f"The directory does not show this requester as entitled to "
                f"{s.data_type.value} data (DCP-2.1 through DCP-2.3).")
    return None


HARD_BLOCK_RULES = [
    _credential_anywhere,
    _personal_device_or_email,
    _removable_media,
    _bulk_external,
    _unsanctioned_ai,
    _not_authorized,
]


def apply_rules(slots: Slots) -> Tuple[Verdict, List[str]]:
    """Return the rule verdict and every reason that fired.

    The rules can only veto. They return DO_NOT_SHARE when a hard block
    fires, and SAFE ("no objection") otherwise. SAFE here is NOT approval:
    agent.decide() combines this with the model verdict using more_cautious(),
    and the model verdict only reaches SAFE when a permissive policy clause
    was retrieved. Approval therefore needs positive evidence from the
    corpus, while a block needs only one rule.

    (An earlier version returned CHECK_POLICY as the no-block default. Since
    the final verdict is the more cautious of the two, that made SAFE
    unreachable for every question. Found by an end-to-end smoke test.)
    """
    reasons: List[str] = []
    for rule in HARD_BLOCK_RULES:
        reason = rule(slots)
        if reason:
            reasons.append(reason)
    if reasons:
        return Verdict.DO_NOT_SHARE, reasons
    return Verdict.SAFE, []
