import json
import os
import torch
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION ---
RUN_TAG = "sbert_dense"
TOP_K = 100
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
OUTPUT_DIR = os.path.join(BASE, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Results_sbert")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
TEST_QRELS_PATH = os.path.join(DATA_DIR, "test.tsv")
QUERIES_PATH = os.path.join(DATA_DIR, "queries.jsonl")

# Using a fast highly accurate model fine-tuned for semantic search
MODEL_NAME = 'all-MiniLM-L6-v2' 

def load_test_query_ids(tsv_path: str) -> set:
    """Extracts the odd-numbered test query IDs from the test.tsv file."""
    qids = set()
    with open(tsv_path, "r", encoding="utf-8") as f:
        next(f) # Skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                qids.add(parts[0].strip())
    return qids

def write_trec_results(out_path: str, run_name: str, ranked_results: List[Tuple[str, List[Tuple[str, float]]]]):
    """Writes the ranked results in the exact TREC format required by trec_eval."""
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, rankings in ranked_results:
            for rank, (doc_id, score) in enumerate(rankings, start=1):
                # Format: query_id Q0 doc_id rank score tag
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

def main():
    print(f"Loading Sentence-BERT model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 1. Load and format the corpus
    print("Loading corpus...")
    corpus_ids = []
    corpus_texts = []
    
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = obj["_id"]
            title = obj.get("title", "")
            text = obj.get("text", "")
            # Combine title and text for a richer semantic representation
            full_text = f"{title} {text}".strip()
            
            corpus_ids.append(doc_id)
            corpus_texts.append(full_text)
            
    # 2. Embed the entire corpus
    print(f"Embedding {len(corpus_texts)} documents (this may take a minute)...")
    # convert_to_tensor=True keeps the embeddings on the GPU if available, speeding up cosine similarity
    corpus_embeddings = model.encode(corpus_texts, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    
    # 3. Load the queries
    print("Loading test queries...")
    test_qids = load_test_query_ids(TEST_QRELS_PATH)
    queries = {}
    
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["_id"])
            if qid in test_qids:
                queries[qid] = obj.get("text", "")
                
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    # 4. Embed the queries
    print(f"Embedding {len(query_texts)} queries...")
    query_embeddings = model.encode(query_texts, batch_size=32, show_progress_bar=True, convert_to_tensor=True)
    
    # 5. Compute Cosine Similarities and extract Top 100
    print("Computing similarities and ranking...")
    all_ranked_results = []
    
    # util.cos_sim computes the cosine similarity of all queries against all documents at once
    cosine_scores = util.cos_sim(query_embeddings, corpus_embeddings)
    
    for i, qid in enumerate(query_ids):
        # Extract the specific row for this query
        query_scores = cosine_scores[i]
        
        # Use PyTorch topk to rapidly grab the top 100 scores and their indices
        top_results = torch.topk(query_scores, k=min(TOP_K, len(corpus_ids)))
        
        ranked_docs = []
        for score, idx in zip(top_results[0], top_results[1]):
            doc_id = corpus_ids[idx]
            ranked_docs.append((str(doc_id), score.item()))
            
        all_ranked_results.append((qid, ranked_docs))
        
    # Sort the final output by query ID ascending to match TREC standard expectations
    all_ranked_results.sort(key=lambda x: int(x[0]))
    
    # 6. Save the results
    write_trec_results(OUTPUT_FILE, RUN_TAG, all_ranked_results)
    print(f"Success! Ranked results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()