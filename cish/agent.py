"""The agent itself: extract_context, decide, log_decision.

Capability note (Requirement 2 of the Part 2 design): neither provider in
llm.py registers a tool schema, so the model has no callable function. It
cannot read files, send mail, or reach any system. The only outbound call in
the whole package is the model request itself, and with CISH_PROVIDER=ollama
even that stays on localhost.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from . import index as policy_index
from . import llm
from . import rules
from .schema import (Channel, DataType, Decision, Provision, Recipient, Slots,
                     Verdict, Volume, more_cautious)

AUDIT_PATH = Path(__file__).resolve().parent.parent / "audit_log.csv"

EXTRACTION_PROMPT = """You extract structured facts from an employee's question about sharing data. You do not judge whether the sharing is allowed.

Return ONLY a JSON object with these keys and no other text:
  data_type: one of customer, employee, financial, credential, internal_doc, public, unknown
  recipient: one of internal, approved_vendor, external, self_personal, public, unknown
  channel:   one of approved_tool, unsanctioned_tool, personal_email, personal_device, removable_media, unknown
  volume:    one of single_record, batch, bulk, unknown

Rules:
- Use "unknown" whenever the question does not clearly state the value. Do not infer.
- Data types: financial is budgets, forecasts, revenue, earnings, financial statements, or payroll totals. employee is salary, pay, performance reviews, or HR records about a person. customer is customer records, accounts, orders, or contact lists. credential is passwords, API keys, tokens, or secrets. internal_doc is ordinary internal material such as meeting notes, plans, or wiki pages. If a document is described by what it contains, classify by that content, not by its form (a photo or screenshot of a forecast is financial).
- A consumer AI service (ChatGPT, public Claude, Gemini, Copilot) is unsanctioned_tool.
- Corporate Google Workspace, corporate Slack, and corporate Jira are approved_tool.
- Sending to one's own personal phone or personal address is self_personal plus the matching channel.
- Volume: one record is single_record. 2 to 99 records, or a handful, is batch. 100 or more records, or any export, database, table, or "thousands" of records, is bulk.
- An outside agency, firm, marketer, or other third party that is not a named approved vendor is external.

Question: {question}"""

VERDICT_PROMPT = """You are advising an employee about a proposed data sharing action. Decide using ONLY the policy clauses provided. Do not use outside knowledge of what is typical or reasonable.

Proposed action:
{context}

Policy clauses retrieved:
{provisions}

Return ONLY a JSON object:
  verdict: one of "SAFE", "CHECK POLICY", "DO NOT SHARE"
  rationale: two sentences at most, citing clause IDs

Rules:
- Answer SAFE only if a clause explicitly permits this action. Absence of a prohibition is not permission.
- Answer DO NOT SHARE if a clause prohibits it.
- Answer CHECK POLICY if the clauses do not clearly settle the case.
"""


def _call_model(prompt: str, max_tokens: int = 400) -> str:
    """Single entry point to whichever provider is configured."""
    return llm.complete(prompt, max_tokens=max_tokens)


def _parse_json(raw: str) -> Optional[dict]:
    """Models occasionally wrap JSON in prose or fences despite instruction."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def extract_context(question: str, requester_id: str,
                    retries: int = 1) -> Slots:
    """Fill the five slots. Never guesses: unfilled slots stay unknown."""
    raw = _call_model(EXTRACTION_PROMPT.format(question=question))
    data = _parse_json(raw)
    if data is None and retries > 0:
        return extract_context(question, requester_id, retries - 1)
    if data is None:
        return Slots(requester_id=requester_id)

    def pick(enum_cls, key, default):
        try:
            return enum_cls(str(data.get(key, "")).strip().lower())
        except ValueError:
            return default

    return Slots(
        data_type=pick(DataType, "data_type", DataType.UNKNOWN),
        recipient=pick(Recipient, "recipient", Recipient.UNKNOWN),
        channel=pick(Channel, "channel", Channel.UNKNOWN),
        volume=pick(Volume, "volume", Volume.UNKNOWN),
        requester_id=requester_id,  # from session, never from the text
    )


def _render_provisions(provisions: List[Provision]) -> str:
    if not provisions:
        return "(no clause in the corpus governs this action)"
    return "\n\n".join(
        f"[{p.clause_id}] {p.heading} ({p.doc_title})\n{p.text}"
        for p in provisions
    )


def model_verdict(slots: Slots, provisions: List[Provision]) -> Verdict:
    if not provisions:
        return Verdict.CHECK_POLICY
    raw = _call_model(VERDICT_PROMPT.format(
        context=slots.as_query(),
        provisions=_render_provisions(provisions),
    ))
    data = _parse_json(raw) or {}
    try:
        verdict = Verdict(str(data.get("verdict", "")).strip().upper())
    except ValueError:
        return Verdict.CHECK_POLICY

    # SAFE requires positive permissive evidence, regardless of what the
    # model concluded. This is the check that prevents a confident model
    # from approving an action the corpus never permitted.
    if verdict == Verdict.SAFE and not \
            policy_index.contains_permissive_clause(provisions):
        return Verdict.CHECK_POLICY

    # Retrieval matches on data type and channel, not recipient, so a clause
    # permitting sharing *inside* the company can surface for a question
    # about sharing *outside* it. SAFE is therefore limited to recipients
    # the permissive clauses actually cover. External, vendor, and personal
    # recipients need a human decision. (Found by test_decide.py.)
    if verdict == Verdict.SAFE and slots.recipient not in (
            Recipient.INTERNAL, Recipient.PUBLIC):
        return Verdict.CHECK_POLICY
    return verdict


def decide(question: str, requester_id: str) -> Decision:
    """Full pipeline. Returns a Decision, possibly a clarifying question."""
    slots = extract_context(question, requester_id)

    # Hard rules run first, on whatever is known. A veto does not need
    # complete information: a bulk export of customer data to an outside
    # party is blocked whether or not the channel was mentioned. Asking a
    # clarifying question before the rules ran let an unmentioned channel
    # downgrade a DO NOT SHARE to CHECK POLICY (found with
    # scripts/diagnose.py on scenario S06).
    rule_v, rule_reasons = rules.apply_rules(slots)

    # Volume is deliberately not required. It only feeds the bulk-external
    # rule, which has already run, and a verdict of SAFE is independently
    # gated on a permissive clause and an internal or public recipient.
    # Requiring it made the answer flip with phrasing (scenario S03).
    missing = [m for m in slots.missing() if m != "volume"]
    if missing and rule_v != Verdict.DO_NOT_SHARE:
        label = {
            "data_type": "what kind of data this is",
            "recipient": "who is receiving it",
            "channel": "what system or device you would use",
        }
        asked = " and ".join(label[m] for m in missing[:2])
        return Decision(
            verdict=Verdict.CHECK_POLICY,
            rationale="More detail is needed before this can be evaluated.",
            slots=slots,
            clarifying_question=f"Before I can answer, could you tell me {asked}?",
        )

    provisions = policy_index.retrieve_policy(slots)

    # When a hard rule has already blocked the request, the final verdict is
    # DO NOT SHARE whatever the model says, because the final verdict is the
    # more cautious of the two. Skipping the call is safe and saves tokens on
    # rate-limited free tiers. It is opt-in because it hides what the model
    # alone would have said, which matters for the model-comparison
    # experiment.
    skip = os.environ.get("CISH_SKIP_MODEL_ON_BLOCK", "").lower() in ("1", "true", "yes")
    if skip and rule_v == Verdict.DO_NOT_SHARE:
        model_v = None
    else:
        model_v = model_verdict(slots, provisions)

    final = rule_v if model_v is None else more_cautious(rule_v, model_v)

    if rule_reasons:
        rationale = " ".join(rule_reasons)
    elif not provisions:
        rationale = ("No clause in the current policy corpus covers this "
                     "action, so it needs a human decision.")
    else:
        rationale = (f"Based on {', '.join(p.clause_id for p in provisions)}, "
                     f"this action is assessed as {final.value}.")

    decision = Decision(
        verdict=final,
        rationale=rationale,
        provisions=provisions,
        slots=slots,
        rule_verdict=rule_v,
        model_verdict=model_v,
    )
    log_decision(question, decision)
    return decision


def log_decision(question: str, decision: Decision) -> None:
    """Append-only audit record. The question is stored; the underlying data
    never is, because the agent never receives it."""
    new_file = not AUDIT_PATH.exists()
    with AUDIT_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["timestamp", "requester_id", "question",
                             "data_type", "recipient", "channel", "volume",
                             "rule_verdict", "model_verdict", "final_verdict",
                             "cited_clauses", "model"])
        s = decision.slots or Slots()
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            s.requester_id, question,
            s.data_type.value, s.recipient.value, s.channel.value,
            s.volume.value,
            decision.rule_verdict.value if decision.rule_verdict else "",
            decision.model_verdict.value if decision.model_verdict else "",
            decision.verdict.value,
            "|".join(decision.citations()),
            llm.active_model(),
        ])
