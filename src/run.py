# src/run.py
import argparse
import json
import os
import sys
from typing import Dict, List, Set, Tuple

from preprocess import load_stopwords
from index import build_index, compute_idf, compute_doc_norms
from rank import rank_query


def load_qrels_qids_from_tsv(test_tsv_path: str) -> Set[str]:
    """
    Reads test.tsv with columns:
      query-id <TAB> corpus-id <TAB> score
    Returns set of query-id strings.
    """
    qids: Set[str] = set()
    with open(test_tsv_path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
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
    """
    Your SciFact queries have fields: _id, text, metadata.
    - mode="title": use first `title_tokens` tokens of query text as a title proxy
    - mode="title_text": use full query text
    """
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
    """
    Writes:
      qid Q0 docid rank score run_name
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for qid, ranked in all_ranked:
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")


def safe_int_qid(qid: str) -> int:
    try:
        return int(float(qid))
    except:
        # Put non-numeric ids at the end
        return 10**18


def main():
    # Your base folder
    BASE = r"C:\Users\smoha\OneDrive - University of Ottawa\Desktop\A1"
    DATA_DIR = os.path.join(BASE, "data")
    OUT_DIR = os.path.join(BASE, "output")

    default_corpus = os.path.join(DATA_DIR, "corpus.jsonl")
    default_queries = os.path.join(DATA_DIR, "queries.jsonl")
    default_test = os.path.join(DATA_DIR, "test.tsv")
    default_stopwords = os.path.join(DATA_DIR, "stopwords.txt")  # rename your stopwords file to this, or pass --stopwords

    ap = argparse.ArgumentParser()

    ap.add_argument("--corpus", default=default_corpus, help="path to corpus.jsonl")
    ap.add_argument("--queries", default=default_queries, help="path to queries.jsonl")
    ap.add_argument("--qrels_tsv", default=default_test, help="path to test.tsv (qrels) used to pick which queries to run")
    ap.add_argument("--stopwords", default=default_stopwords, help="path to stopwords .txt")

    ap.add_argument("--mode", choices=["title", "title_text"], default="title_text",
                    help="title=first N tokens of query text (title proxy), title_text=full query text")
    ap.add_argument("--title_tokens", type=int, default=10, help="N tokens for title proxy (only used with --mode title)")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--run_name", default=None, help="tag in Results lines (same for all lines). If omitted, auto-set.")
    ap.add_argument("--out", default=None, help="output file path. If omitted, auto-set under A1/output/.")

    ap.add_argument("--do_two_runs", action="store_true",
                    help="If set, produces BOTH runs: title-proxy and full-text, into A1/output/.")

    args = ap.parse_args()

    # Basic file existence checks (friendly errors)
    for p, label in [
        (args.corpus, "corpus.jsonl"),
        (args.queries, "queries.jsonl"),
        (args.qrels_tsv, "test.tsv (qrels)"),
        (args.stopwords, "stopwords.txt"),
    ]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing {label} at:\n  {p}")
            sys.exit(1)

    stopwords: Set[str] = load_stopwords(args.stopwords)

    print("Building index from corpus...")
    index, doc_len, N, vocab = build_index(args.corpus, stopwords)
    print(f"Docs indexed: {N}")
    print(f"Vocabulary size: {len(vocab)}")

    print("Computing IDF + document norms...")
    idf = compute_idf(index, N)
    doc_norms = compute_doc_norms(index, idf)

    print("Loading queries + qrels ids...")
    queries = load_queries(args.queries)
    qrels_qids = load_qrels_qids_from_tsv(args.qrels_tsv)

    # Filter queries to ONLY those appearing in qrels
    filtered: List[Tuple[str, Dict]] = []
    for q in queries:
        qid = q.get("_id")
        if qid is None:
            continue
        qid = str(qid)
        if qid in qrels_qids:
            filtered.append((qid, q))

    filtered.sort(key=lambda x: safe_int_qid(x[0]))
    print(f"Queries selected (present in qrels): {len(filtered)}")

    def run_one(mode: str, title_tokens: int) -> str:
        run_name = args.run_name
        if not run_name:
            run_name = "tfidf_title" + str(title_tokens) if mode == "title" else "tfidf_full"

        out_path = args.out
        if not out_path:
            fname = "Results_title" + str(title_tokens) if mode == "title" else "Results_full"
            out_path = os.path.join(OUT_DIR, fname)

        all_ranked: List[Tuple[str, List[Tuple[str, float]]]] = []
        for qid, qobj in filtered:
            qtext = get_query_text(qobj, mode, title_tokens=title_tokens)
            ranked = rank_query(qtext, index, idf, doc_norms, stopwords, topk=args.topk)
            all_ranked.append((qid, ranked))

        write_trec_results(out_path, run_name, all_ranked)
        print(f"Wrote: {out_path}")
        return out_path

    if args.do_two_runs:
        # Run A: title proxy
        run_one(mode="title", title_tokens=args.title_tokens)
        # Run B: full query text
        # For second run we should not overwrite args.out if user provided it;
        # but if user provided --out, they probably only want one file.
        # We'll force output names under OUT_DIR for the two-runs mode.
        old_out = args.out
        old_run = args.run_name
        args.out = None
        args.run_name = None
        run_one(mode="title_text", title_tokens=args.title_tokens)
        args.out = old_out
        args.run_name = old_run
    else:
        run_one(mode=args.mode, title_tokens=args.title_tokens)

    print("Done.")


if __name__ == "__main__":
    main()
