# src/rank.py
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set

from preprocess import preprocess
from index import IndexType
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer, util

# Load neural model once
model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# cache for document embeddings so we compute them only once
doc_embedding_cache: Dict[str, object] = {}


def rank_query(
    query_text: str,
    index: IndexType,
    idf: Dict[str, float],
    doc_norms: Dict[str, float],
    stopwords: Set[str],
    topk: int = 100
) -> List[Tuple[str, float]]:
    """
    Returns top-k list of (doc_id, score) sorted by score desc.
    Baseline TF-IDF cosine ranking (Assignment 1).
    """

    q_tokens = preprocess(query_text, stopwords)
    if not q_tokens:
        return []

    q_tf = Counter(q_tokens)

    # query weights and norm
    q_weights: Dict[str, float] = {}
    q_sum_sq = 0.0

    for term, tf in q_tf.items():
        if term not in idf:
            continue

        w = tf * idf[term]
        q_weights[term] = w
        q_sum_sq += w * w

    q_norm = math.sqrt(q_sum_sq)

    if q_norm == 0.0:
        return []

    # dot products over candidate docs in postings
    dot = defaultdict(float)

    for term, wq in q_weights.items():
        postings = index.get(term)

        if not postings:
            continue

        w_idf = idf[term]

        for doc_id, tf_d in postings.items():
            wd = tf_d * w_idf
            dot[doc_id] += wq * wd

    scored = []

    for doc_id, numerator in dot.items():

        d_norm = doc_norms.get(doc_id, 0.0)

        if d_norm == 0.0:
            continue

        score = numerator / (d_norm * q_norm)

        scored.append((doc_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:topk]


def neural_rerank(
    query_text: str,
    ranked_docs: List[Tuple[str, float]],
    corpus_lookup: Dict[str, str],
    topk: int = 100
) -> List[Tuple[str, float]]:
    """
    Neural reranking using Sentence-BERT embeddings (Assignment 2).
    """

    query_emb = model.encode(query_text, convert_to_tensor=True)

    reranked = []

    for doc_id, _ in ranked_docs:

        doc_text = corpus_lookup.get(doc_id, "")

        # use cached embedding if available
        if doc_id not in doc_embedding_cache:
            doc_embedding_cache[doc_id] = model.encode(
                doc_text,
                convert_to_tensor=True
            )

        doc_emb = doc_embedding_cache[doc_id]

        score = util.cos_sim(query_emb, doc_emb).item()

        reranked.append((doc_id, score))

    reranked.sort(key=lambda x: x[1], reverse=True)

    return reranked[:topk]



def neural_rerank_cross_encoder(
    query_text: str,
    ranked_docs: List[Tuple[str, float]],
    corpus_lookup: Dict[str, str],
    topk: int = 100
) -> List[Tuple[str, float]]:

    pairs = []
    doc_ids = []

    for doc_id, _ in ranked_docs:
        doc_text = corpus_lookup.get(doc_id, "")
        pairs.append((query_text, doc_text))
        doc_ids.append(doc_id)

    scores = cross_encoder.predict(pairs)

    reranked = list(zip(doc_ids, scores))

    reranked.sort(key=lambda x: x[1], reverse=True)

    return reranked[:topk]