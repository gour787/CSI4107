# src/rank.py
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set

from preprocess import preprocess
from index import IndexType

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
    """
    q_tokens = preprocess(query_text, stopwords)
    if not q_tokens:
        return []

    q_tf = Counter(q_tokens)

    # Build query weights and norm
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

    # Accumulate dot products only over candidate docs in postings
    dot = defaultdict(float)  # doc_id -> dot(d,q)
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
