"""Policy corpus parsing, plus the build_index / retrieve_policy entry points.

Parsing lives here; the actual scoring lives in retrieval.py behind a
swappable backend. Callers only ever touch retrieve_policy(), so changing
how retrieval works does not touch the agent, the rules engine, or the UI.

Chunking note: the Part 2 design specified a 500-token recursive splitter.
We chunk on clause headings instead, because policy clauses are short and
already semantically bounded, and the fixed-size splitter was cutting them
in half. This is a deliberate deviation worth reporting.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from . import retrieval
from .schema import Provision, Slots

POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"

HEADING_RE = re.compile(r"^###\s+(\S+)\s+(.*)$")
META_RE = re.compile(r"^<!--\s*(.*?)\s*-->$")

# An empty provision list is a legitimate, common outcome, not an error.
# It is the main path by which a question reaches CHECK POLICY.
NO_GOVERNING_PROVISION: List[Provision] = []


def parse_policy_file(path: Path) -> List[dict]:
    """Split one markdown policy document into tagged clause chunks."""
    lines = path.read_text(encoding="utf-8").splitlines()
    doc_title = "Untitled Policy"
    doc_version = "unknown"
    chunks: List[dict] = []
    current: Optional[dict] = None

    for line in lines:
        if line.startswith("# "):
            doc_title = line[2:].strip()
            continue
        if line.lower().startswith("version "):
            doc_version = line.strip()
            continue

        heading = HEADING_RE.match(line)
        if heading:
            if current:
                chunks.append(current)
            current = {
                "clause_id": heading.group(1),
                "heading": heading.group(2).strip(),
                "text": "",
                "doc_title": doc_title,
                "doc_version": doc_version,
                "data_types": "",
                "channels": "",
                "disposition": "restrictive",
            }
            continue

        meta = META_RE.match(line.strip())
        if meta and current:
            for part in meta.group(1).split(";"):
                if ":" not in part:
                    continue
                key, value = part.split(":", 1)
                key, value = key.strip(), value.strip()
                if key in ("data_types", "channels", "disposition"):
                    current[key] = value
            continue

        if current and line.strip():
            current["text"] += (" " if current["text"] else "") + line.strip()

    if current:
        chunks.append(current)

    for ch in chunks:
        ch["doc_title"] = doc_title
        ch["doc_version"] = doc_version
    return chunks


def load_corpus(policy_dir: Path = POLICY_DIR) -> List[dict]:
    chunks: List[dict] = []
    for path in sorted(policy_dir.glob("*.md")):
        chunks.extend(parse_policy_file(path))
    return chunks


def build_index(policy_dir: Path = POLICY_DIR) -> int:
    """Build whichever retrieval backend is configured."""
    retrieval.reset_cache()
    chunks = load_corpus(policy_dir)
    if retrieval.backend_name() == "tfidf":
        # TF-IDF is cheap enough to fit at process start, so there is no
        # persisted artifact to build. Returning the count keeps the CLI
        # output identical across backends.
        retrieval.get_retriever()
        return len(chunks)
    backend = retrieval.EmbeddingRetriever()
    return backend.build(chunks)


def retrieve_policy(slots: Slots, k: int = 4) -> List[Provision]:
    return retrieval.get_retriever().retrieve(slots, k=k)


def contains_permissive_clause(provisions: List[Provision]) -> bool:
    """SAFE requires positive permission, not merely the absence of a ban."""
    return any(p.disposition == "permissive" for p in provisions)


if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} clauses using the "
          f"'{retrieval.backend_name()}' backend.")
