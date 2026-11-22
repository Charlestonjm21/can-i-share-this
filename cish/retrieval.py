"""Retrieval backends.

Two implementations behind one interface, selected with CISH_RETRIEVER:

    tfidf       keyword scoring over the clause corpus   (default, no model)
    embedding   sentence-transformers + Chroma           (optional extra)

Why TF-IDF is the default
-------------------------
The Part 2 design specified dense embeddings. During implementation we
noticed that the retriever never sees the employee's own words. By the time
retrieve() runs, agent.extract_context() has already normalised the question
into enum values, and Slots.as_query() renders those enums into a fixed
vocabulary sentence:

    "Sharing customer data with a external recipient via unsanctioned_tool,
     volume bulk."

So retrieval is matching a controlled vocabulary against clauses tagged with
that same controlled vocabulary. The paraphrase robustness that justifies
dense embeddings was already supplied one stage earlier, by the extractor.
Keyword scoring is a good fit for that, and it removes a ~2 GB PyTorch
dependency from the default install.

Both backends are kept so the choice can be measured rather than asserted:
run evaluate.py under each and compare.
"""

from __future__ import annotations

import os
from typing import List, Protocol

from .schema import Provision, Slots

RELEVANCE_FLOOR = float(os.environ.get("CISH_RELEVANCE_FLOOR", "0.25"))

# Scoring weights. Metadata agreement dominates, text similarity breaks ties.
#
# SPECIFICITY is the interesting one. A clause tagged for one data type
# (AUP-4.1, credentials only) targets this case far more precisely than a
# clause tagged for all six (DCP-1.1, the general classification scheme).
# Both "match" on tags, so without a specificity term the general clause wins
# on vocabulary overlap and the governing clause never surfaces. Ranking
# narrow clauses above broad ones is also how a lawyer reads a policy: the
# specific provision controls over the general one.
W_TYPE = 0.30
W_CHANNEL = 0.15
W_SPECIFICITY = 0.35
W_SIMILARITY = 0.20


def backend_name() -> str:
    return os.environ.get("CISH_RETRIEVER", "tfidf").strip().lower()


class Retriever(Protocol):
    def build(self, chunks: List[dict]) -> int: ...
    def retrieve(self, slots: Slots, k: int = 4) -> List[Provision]: ...


def _apply_metadata(chunk: dict, slots: Slots, similarity: float):
    """Score a candidate against the slots. Returns None to drop it.

    similarity is expected in roughly 0..1; it contributes least because a
    twenty-clause corpus gives TF-IDF little to work with.
    """
    types = [t.strip() for t in chunk.get("data_types", "").split(",") if t.strip()]
    channels = [c.strip() for c in chunk.get("channels", "").split(",") if c.strip()]
    type_hit = slots.data_type.value in types
    channel_hit = slots.channel.value in channels
    if not (type_hit or channel_hit):
        return None

    # How narrowly does this clause target what was asked about? A clause
    # listing one data type scores 1.0; one listing all six scores 0.17.
    specificity = 0.0
    if type_hit:
        specificity = max(specificity, 1.0 / len(types))
    if channel_hit:
        specificity = max(specificity, 1.0 / len(channels))

    return (
        W_TYPE * float(type_hit)
        + W_CHANNEL * float(channel_hit)
        + W_SPECIFICITY * specificity
        + W_SIMILARITY * min(similarity, 1.0)
    )


def _to_provision(chunk: dict, score: float) -> Provision:
    return Provision(
        clause_id=chunk["clause_id"],
        heading=chunk["heading"],
        text=chunk["text"],
        doc_title=chunk["doc_title"],
        doc_version=chunk["doc_version"],
        disposition=chunk.get("disposition", "restrictive"),
        score=round(score, 4),
    )


class TfidfRetriever:
    """Term frequency / inverse document frequency over the clause corpus.

    Rare terms carry the ranking: "credential" appears in two clauses and
    weighs heavily, "the" appears in all twenty and weighs nothing.
    """

    def __init__(self):
        self.chunks: List[dict] = []
        self.vectorizer = None
        self.matrix = None

    def build(self, chunks: List[dict]) -> int:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        corpus = [
            # Tags are folded into the indexed text so that a clause tagged
            # for unsanctioned_tool scores on a query naming that channel,
            # even when its prose says "consumer AI services" instead.
            f"{c['heading']}. {c['text']} "
            f"{c.get('data_types', '')} {c.get('channels', '')}"
            for c in chunks
        ]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            token_pattern=r"[A-Za-z_][A-Za-z_]+",
            ngram_range=(1, 2),
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)
        return len(chunks)

    def retrieve(self, slots: Slots, k: int = 4) -> List[Provision]:
        from sklearn.metrics.pairwise import cosine_similarity

        if self.vectorizer is None:
            self.build(_load_chunks())

        query_vec = self.vectorizer.transform([slots.as_retrieval_query()])
        sims = cosine_similarity(query_vec, self.matrix)[0]

        scored = []
        for chunk, sim in zip(self.chunks, sims):
            score = _apply_metadata(chunk, slots, float(sim))
            if score is None:
                continue
            scored.append(_to_provision(chunk, score))

        scored.sort(key=lambda p: p.score, reverse=True)
        top = scored[:k]
        if not top or top[0].score < RELEVANCE_FLOOR:
            return []
        return top


class EmbeddingRetriever:
    """sentence-transformers + Chroma. Requires the optional extras."""

    COLLECTION = "policy_corpus"
    MODEL = os.environ.get("CISH_EMBED_MODEL", "all-MiniLM-L6-v2")

    def __init__(self, persist_dir=None):
        from pathlib import Path
        self.persist_dir = str(
            persist_dir or Path(__file__).resolve().parent.parent / ".chroma"
        )

    def _collection(self, create=False):
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path=self.persist_dir)
        fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.MODEL
        )
        if create:
            try:
                client.delete_collection(self.COLLECTION)
            except Exception:
                pass
            return client.get_or_create_collection(
                name=self.COLLECTION, embedding_function=fn
            )
        return client.get_collection(name=self.COLLECTION,
                                     embedding_function=fn)

    def build(self, chunks: List[dict]) -> int:
        collection = self._collection(create=True)
        collection.add(
            ids=[c["clause_id"] for c in chunks],
            documents=[f"{c['heading']}. {c['text']}" for c in chunks],
            metadatas=[{k: c.get(k, "") for k in (
                "clause_id", "heading", "doc_title", "doc_version",
                "data_types", "channels", "disposition", "text")}
                for c in chunks],
        )
        return len(chunks)

    def retrieve(self, slots: Slots, k: int = 4) -> List[Provision]:
        try:
            collection = self._collection()
        except Exception as exc:
            raise RuntimeError(
                "Embedding index not found. Run `python -m cish.index` with "
                "CISH_RETRIEVER=embedding, and install the extras from "
                "requirements-embeddings.txt."
            ) from exc

        result = collection.query(query_texts=[slots.as_retrieval_query()],
                                  n_results=k * 3)
        scored = []
        for meta, dist in zip(result["metadatas"][0], result["distances"][0]):
            base = 1.0 / (1.0 + dist)
            score = _apply_metadata(meta, slots, base)
            if score is None:
                continue
            scored.append(_to_provision(meta, score))

        scored.sort(key=lambda p: p.score, reverse=True)
        top = scored[:k]
        if not top or top[0].score < RELEVANCE_FLOOR:
            return []
        return top


_ACTIVE: dict = {}


def _load_chunks():
    from .index import load_corpus
    return load_corpus()


def get_retriever() -> Retriever:
    """Return the configured backend, built and cached for this process."""
    name = backend_name()
    if name in _ACTIVE:
        return _ACTIVE[name]

    if name == "tfidf":
        retriever = TfidfRetriever()
        retriever.build(_load_chunks())
    elif name == "embedding":
        retriever = EmbeddingRetriever()
    else:
        raise RuntimeError(
            f"Unknown CISH_RETRIEVER '{name}'. Use 'tfidf' or 'embedding'."
        )

    _ACTIVE[name] = retriever
    return retriever


def reset_cache() -> None:
    """Used by tests and by evaluate.py --compare when switching backends."""
    _ACTIVE.clear()
