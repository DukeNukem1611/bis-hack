import json
import argparse
import time
import pickle
import torch
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# 1. LOAD MODELS ONCE GLOBALLY
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', 
                        activation_fn=torch.nn.Sigmoid(), 
                        device=device)

def load_knowledge_base(pkl_path):
    """Loads the pre-computed chunks, metadata, and embeddings from disk."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['corpus'], data['metadata'], torch.tensor(data['embeddings'], device=device)

def normalize_standard_id(std_id):
    clean_id = re.sub(r'\s+', ' ', std_id).strip()
    match = re.search(r'IS\s*(\d+)\s*(?:\(\s*Part\s*(\d+)\s*\))?\s*:\s*(\d{4})', clean_id)
    if match:
        number = match.group(1)
        part = match.group(2)
        year = match.group(3)
        if part:
            return f"IS {number} (Part {part}): {year}"
        else:
            return f"IS {number}: {year}"
    return clean_id

def main(input_path, output_path):
    # 2. LOAD PRE-COMPUTED INDEXES (Latency saver!)
    # Update this path relative to where the eval script calls inference.py
    corpus, metadata, corpus_embeddings = load_knowledge_base('data/knowledge_base.pkl')
    
    # Initialize BM25 with the loaded corpus
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    results = []
    
    # 3. READ INPUT DATASET
    with open(input_path, 'r') as f:
        queries = json.load(f)

    # 4. PROCESS QUERIES
    for item in queries:
        start_time = time.time()
        query_text = item.get("query", "")

        # --- A. HYBRID RETRIEVAL ---
        
        # 1. Dense Retrieval ONLY
        query_embedding = embedder.encode(query_text, convert_to_tensor=True, device=device)
        dense_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=20)[0]
        top_n_dense_idx = [hit['corpus_id'] for hit in dense_hits]

        combined_indices = top_n_dense_idx

        # --- B. CROSS-ENCODER RERANKING ---
        cross_inp = [[query_text, corpus[idx]] for idx in combined_indices]
        
        # We use a try-except block just in case cross_inp is empty (malformed query)
        if cross_inp:
            cross_scores = reranker.predict(cross_inp)
            ranked_pairs = sorted(zip(combined_indices, cross_scores), key=lambda x: x[1], reverse=True)
        else:
            ranked_pairs = []

# --- C. METADATA EXTRACTION (AGGREGATING SCORES) ---
        standard_scores = {}
        for rank, (idx, score) in enumerate(ranked_pairs):
            std_id = normalize_standard_id(metadata[idx]["Standard_ID"])
            
            # Weighted scoring based on Cross Encoder rank.
            # Using overlapping chunks means we might get multiple highly-scored chunks for the same standard.
            if std_id not in standard_scores:
                standard_scores[std_id] = float(score)
            else:
                standard_scores[std_id] += float(score) * 0.1 
                
        # Sort standards by their aggregated cross-encoder scores
        sorted_standards = sorted(standard_scores.items(), key=lambda x: x[1], reverse=True)
        retrieved_standards = [std_id for std_id, score in sorted_standards][:5]

        # Record latency for the evaluation script
        latency = time.time() - start_time

        # Ensure exact match with Hackathon JSON schema
        results.append({
            "id": item["id"],
            "retrieved_standards": retrieved_standards,
            "latency_seconds": round(latency, 4)
        })

    # 5. WRITE OUTPUT DATASET
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIS Hackathon Inference Script")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON")
    args = parser.parse_args()
    
    main(args.input, args.output)