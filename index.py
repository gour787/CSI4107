# src/index.py
import json
import math
from collections import Counter, defaultdict
from typing import Dict, Tuple, Set

from preprocess import preprocess

IndexType = Dict[str, Dict[str, int]]  # term -> {doc_id -> tf}

def build_index(corpus_path: str, stopwords: Set[str]) -> Tuple[IndexType, Dict[str, int], int, Set[str]]:
    """
    Builds inverted index with raw TF.
    Also returns doc_len, N, vocab.
    """
    index: IndexType = defaultdict(dict)
    doc_len: Dict[str, int] = {}
    vocab: Set[str] = set()

    N = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = obj["_id"]
            title = obj.get("title", "") or ""
            text = obj.get("text", "") or ""
            full_text = f"{title} {text}"

            tokens = preprocess(full_text, stopwords)
            doc_len[doc_id] = len(tokens)
            N += 1

            tf_counts = Counter(tokens)
            for term, tf in tf_counts.items():
                index[term][doc_id] = tf
                vocab.add(term)

    return dict(index), doc_len, N, vocab

def compute_idf(index: IndexType, N: int) -> Dict[str, float]:
    """
    Smoothed IDF:
      idf = log((N+1)/(df+1)) + 1
    """
    idf: Dict[str, float] = {}
    for term, postings in index.items():
        df = len(postings)
        idf[term] = math.log((N + 1) / (df + 1)) + 1.0
    return idf

def compute_doc_norms(index: IndexType, idf: Dict[str, float]) -> Dict[str, float]:
    """
    Precompute ||d|| for cosine similarity:
      ||d|| = sqrt(sum_t (tf(t,d)*idf(t))^2)
    """
    accum = defaultdict(float)  # doc_id -> sum of squares
    for term, postings in index.items():
        w_idf = idf.get(term, 0.0)
        if w_idf == 0.0:
            continue
        for doc_id, tf in postings.items():
            w = tf * w_idf
            accum[doc_id] += w * w

    doc_norms = {doc_id: math.sqrt(s) for doc_id, s in accum.items()}
    return doc_norms
