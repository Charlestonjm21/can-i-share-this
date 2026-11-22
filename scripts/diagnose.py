"""Show what the model extracts for each phrasing of one scenario.

    python scripts/diagnose.py S06

Prints the raw model output, the slots it was parsed into, the rule result,
and the final verdict, so you can see WHERE a wrong answer comes from.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from cish import agent, rules


def main(sid):
    scenarios = {s["id"]: s for s in yaml.safe_load(open("scenarios.yaml"))}
    s = scenarios[sid]
    print(f"{sid} requester={s['requester']} expected={s['expected']}\n")
    for q in [s["question"]] + list(s.get("paraphrases", [])):
        raw = agent._call_model(agent.EXTRACTION_PROMPT.format(question=q))
        slots = agent.extract_context(q, s["requester"])
        rv, reasons = rules.apply_rules(slots)
        d = agent.decide(q, s["requester"])
        print("Q:    ", q)
        print("RAW:  ", raw.strip().replace("\n", " ")[:300])
        print("SLOTS:", slots.as_query())
        print("RULES:", rv.value, reasons)
        print("FINAL:", d.verdict.value, "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "S06")
