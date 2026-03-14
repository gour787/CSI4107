# src/run.py
import argparse
import json
import os
import sys
from typing import Dict, List, Set, Tuple

from preprocess import load_stopwords
from index import build_index, compute_idf, compute_doc_norms
from rank import rank_query, neural_rerank, neural_rerank_cross_encoder


def load_corpus_lookup(corpus_path: str) -> Dict[str, str]:
    """
    Loads corpus text so neural model can access document text.
    """
    lookup = {}

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            doc_id = obj["_id"]
            title = obj.get("title", "")
            text = obj.get("text", "")

            lookup[doc_id] = title + " " + text

    return lookup


def load_qrels_qids_from_tsv(test_tsv_path: str) -> Set[str]:
    qids: Set[str] = set()

    with open(test_tsv_path, "r", encoding="utf-8") as f:
        f.readline()

        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 2:
                continue

            qid = parts[0].strip()

            if qid:
                qids.add(qid)

    return qids


def load_queries(queries_path: str) -> List[Dict]:

    queries = []

    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                queries.append(json.loads(line))

    return queries


def get_query_text(qobj: Dict, mode: str, title_tokens: int = 10) -> str:

    text = (qobj.get("text", "") or "").strip()

    if mode == "title":
        parts = text.split()
        return " ".join(parts[:title_tokens])

    return text


def write_trec_results(
    out_path: str,
    run_name: str,
    all_ranked: List[Tuple[str, List[Tuple[str, float]]]]
) -> None:

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out:

        for qid, ranked in all_ranked:

            for rank, (doc_id, score) in enumerate(ranked, start=1):

                out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")


def safe_int_qid(qid: str) -> int:
    try:
        return int(float(qid))
    except:
        return 10**18


def main():

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE, "data")
    OUT_DIR = os.path.join(BASE, "output")

    default_corpus = os.path.join(DATA_DIR, "corpus.jsonl")
    default_queries = os.path.join(DATA_DIR, "queries.jsonl")
    default_test = os.path.join(DATA_DIR, "test.tsv")
    default_stopwords = os.path.join(DATA_DIR, "stopwords.txt")

    ap = argparse.ArgumentParser()

    ap.add_argument("--corpus", default=default_corpus)
    ap.add_argument("--queries", default=default_queries)
    ap.add_argument("--qrels_tsv", default=default_test)
    ap.add_argument("--stopwords", default=default_stopwords)

    ap.add_argument("--mode", choices=["title", "title_text"], default="title_text")
    ap.add_argument("--title_tokens", type=int, default=10)

    ap.add_argument("--topk", type=int, default=100)

    args = ap.parse_args()

    for p in [args.corpus, args.queries, args.qrels_tsv, args.stopwords]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing file: {p}")
            sys.exit(1)

    stopwords: Set[str] = load_stopwords(args.stopwords)

    print("Building index...")
    index, doc_len, N, vocab = build_index(args.corpus, stopwords)

    print("Computing IDF...")
    idf = compute_idf(index, N)

    print("Computing doc norms...")
    doc_norms = compute_doc_norms(index, idf)

    print("Loading corpus lookup...")
    corpus_lookup = load_corpus_lookup(args.corpus)

    print("Loading queries...")
    queries = load_queries(args.queries)

    qrels_qids = load_qrels_qids_from_tsv(args.qrels_tsv)

    filtered: List[Tuple[str, Dict]] = []

    for q in queries:

        qid = q.get("_id")

        if qid is None:
            continue

        qid = str(qid)

        if qid in qrels_qids:
            filtered.append((qid, q))

    filtered.sort(key=lambda x: safe_int_qid(x[0]))

    print(f"Queries selected: {len(filtered)}")

    tfidf_results = []
    sbert_results = []
    cross_results = []

    for qid, qobj in filtered:

        qtext = get_query_text(qobj, args.mode, title_tokens=args.title_tokens)

        # TF-IDF baseline
        tfidf_rank = rank_query(
            qtext,
            index,
            idf,
            doc_norms,
            stopwords,
            topk=args.topk
        )

        # Sentence-BERT reranking
        sbert_rank = neural_rerank(
            qtext,
            tfidf_rank[:50],
            corpus_lookup,
            topk=args.topk
        )

        # Cross-Encoder reranking
        cross_rank = neural_rerank_cross_encoder(
            qtext,
            tfidf_rank[:30],
            corpus_lookup,
            topk=args.topk
        )

        tfidf_results.append((qid, tfidf_rank))
        sbert_results.append((qid, sbert_rank))
        cross_results.append((qid, cross_rank))

    write_trec_results(
        os.path.join(OUT_DIR, "Results_tfidf"),
        "tfidf",
        tfidf_results
    )

    write_trec_results(
        os.path.join(OUT_DIR, "Results_sbert"),
        "sbert",
        sbert_results
    )

    write_trec_results(
        os.path.join(OUT_DIR, "Results_cross"),
        "cross_encoder",
        cross_results
    )

    print("Generated result files:")
    print("Results_tfidf")
    print("Results_sbert")
    print("Results_cross")


if __name__ == "__main__":
    main()