"""Corpus parsing tests.

No API key and no model needed, so these run in CI on every push.
"""

import pytest

from cish.index import load_corpus, parse_policy_file, POLICY_DIR
from cish.schema import Channel, DataType

VALID_TYPES = {d.value for d in DataType}
VALID_CHANNELS = {c.value for c in Channel}


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def test_corpus_is_not_empty(corpus):
    assert len(corpus) >= 15


def test_every_clause_has_an_id_and_text(corpus):
    for c in corpus:
        assert c["clause_id"], "clause with no id"
        assert len(c["text"]) > 40, f"{c['clause_id']} has almost no text"


def test_clause_ids_are_unique(corpus):
    ids = [c["clause_id"] for c in corpus]
    assert len(ids) == len(set(ids)), "duplicate clause id"


def test_tags_use_the_enum_vocabulary(corpus):
    """A typo in a policy tag silently removes that clause from retrieval,
    which is the kind of bug that is invisible until a verdict is wrong."""
    for c in corpus:
        for t in [x.strip() for x in c["data_types"].split(",") if x.strip()]:
            assert t in VALID_TYPES, f"{c['clause_id']}: bad data type '{t}'"
        for ch in [x.strip() for x in c["channels"].split(",") if x.strip()]:
            assert ch in VALID_CHANNELS, f"{c['clause_id']}: bad channel '{ch}'"


def test_dispositions_are_known(corpus):
    allowed = {"permissive", "restrictive", "prohibitive"}
    for c in corpus:
        assert c["disposition"] in allowed, \
            f"{c['clause_id']}: bad disposition '{c['disposition']}'"


def test_corpus_has_at_least_one_permissive_clause(corpus):
    """Without one, SAFE is unreachable and the agent can only ever warn."""
    assert any(c["disposition"] == "permissive" for c in corpus)


def test_each_policy_file_parses(tmp_path):
    for path in POLICY_DIR.glob("*.md"):
        chunks = parse_policy_file(path)
        assert chunks, f"{path.name} produced no clauses"
        assert all(c["doc_title"] != "Untitled Policy" for c in chunks), \
            f"{path.name} is missing its top-level '# Title' heading"
