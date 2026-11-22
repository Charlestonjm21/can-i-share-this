"""Data models for the Can I Share This? agent.

Implements the SLOT_SCHEMA and VerdictSchema from the Part 2 design document.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DataType(str, Enum):
    CUSTOMER = "customer"
    EMPLOYEE = "employee"
    FINANCIAL = "financial"
    CREDENTIAL = "credential"
    INTERNAL_DOC = "internal_doc"
    PUBLIC = "public"
    UNKNOWN = "unknown"


class Recipient(str, Enum):
    INTERNAL = "internal"
    APPROVED_VENDOR = "approved_vendor"
    EXTERNAL = "external"
    SELF_PERSONAL = "self_personal"
    PUBLIC = "public"
    UNKNOWN = "unknown"


class Channel(str, Enum):
    APPROVED_TOOL = "approved_tool"
    UNSANCTIONED_TOOL = "unsanctioned_tool"
    PERSONAL_EMAIL = "personal_email"
    PERSONAL_DEVICE = "personal_device"
    REMOVABLE_MEDIA = "removable_media"
    UNKNOWN = "unknown"


class Volume(str, Enum):
    SINGLE_RECORD = "single_record"
    BATCH = "batch"
    BULK = "bulk"
    UNKNOWN = "unknown"


class Verdict(str, Enum):
    SAFE = "SAFE"
    CHECK_POLICY = "CHECK POLICY"
    DO_NOT_SHARE = "DO NOT SHARE"


# Ascending caution. decide() takes the maximum index of the two verdicts,
# which is what makes every stage able only to tighten the recommendation.
SAFETY_ORDER: List[Verdict] = [Verdict.SAFE, Verdict.CHECK_POLICY, Verdict.DO_NOT_SHARE]


def more_cautious(a: Verdict, b: Verdict) -> Verdict:
    return SAFETY_ORDER[max(SAFETY_ORDER.index(a), SAFETY_ORDER.index(b))]


class Slots(BaseModel):
    """The five contextual-integrity factors, plus volume."""

    data_type: DataType = DataType.UNKNOWN
    recipient: Recipient = Recipient.UNKNOWN
    channel: Channel = Channel.UNKNOWN
    volume: Volume = Volume.UNKNOWN
    requester_id: str = ""

    def missing(self) -> List[str]:
        """Slots the extractor could not fill. requester_id is excluded
        because it comes from the session, not from the question text."""
        out = []
        for name in ("data_type", "recipient", "channel", "volume"):
            if getattr(self, name).value == "unknown":
                out.append(name)
        return out

    def as_query(self) -> str:
        """Human-readable rendering, used in the model prompt."""
        return (
            f"Sharing {self.data_type.value} data "
            f"with a {self.recipient.value} recipient "
            f"via {self.channel.value}, volume {self.volume.value}."
        )

    def as_retrieval_query(self) -> str:
        """Bare tokens for the retriever.

        The sentence form above carries filler ("sharing", "data", "with",
        "recipient") that appears in nearly every policy clause and swamps
        the signal terms under TF-IDF. The retriever gets the enum values
        alone.
        """
        return " ".join([
            self.data_type.value,
            self.recipient.value,
            self.channel.value,
            self.volume.value,
        ])


class Provision(BaseModel):
    """One retrieved policy clause."""

    clause_id: str
    heading: str
    text: str
    doc_title: str
    doc_version: str
    disposition: str  # permissive | restrictive | prohibitive
    score: float = 0.0


class Decision(BaseModel):
    verdict: Verdict
    rationale: str
    provisions: List[Provision] = Field(default_factory=list)
    slots: Optional[Slots] = None
    clarifying_question: Optional[str] = None
    rule_verdict: Optional[Verdict] = None
    model_verdict: Optional[Verdict] = None

    def citations(self) -> List[str]:
        return [p.clause_id for p in self.provisions]
