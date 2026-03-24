"""
evaluate_rag.py — RAG Pipeline Evaluation

Usage:
    python evaluate_rag.py --payload output_payload.json --test-cases test_cases.json

    # Retrieval or generation only
    python evaluate_rag.py ... --mode retrieval
    python evaluate_rag.py ... --mode generation

Input schemas:
    output_payload.json   {"results": [{"query_id", "query", "response", "retrieved_context": [{"doc_id", "text"}]}]}
    test_cases.json       [{"id": 1, "reference": "..."}]  ← ids are 1-indexed

Requirements:
    pip install sentence-transformers scikit-learn numpy bert-score tqdm
"""

import argparse, json, os, re, sys
import numpy as np
from typing import Dict

STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","of","and","or","but","for",
    "with","from","that","this","was","are","be","as","by","its","where","what",
    "how","did","does","do","has","have","been","not","which","who","they",
    "their","also","into","can",
}


def load_references(path: str) -> Dict[int, str]:
    test_cases = json.load(open(path))
    return {tc["id"] - 1: tc["reference"] for tc in test_cases}


def evaluate_retrieval(results, refs, model_name):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer(model_name)
    recalls, semantics = [], []

    for item in results:
        qid = int(item["query_id"])
        if qid not in refs:
            continue

        ref      = refs[qid]
        context  = " ".join(c["text"].lower() for c in item["retrieved_context"])
        keywords = [t for t in re.findall(r"\b[a-zA-Z]{3,}\b", ref.lower()) if t not in STOPWORDS]
        hits     = [kw for kw in keywords if kw in context]
        recall   = len(hits) / len(keywords) if keywords else 0.0
        semantic = float(cosine_similarity(model.encode([ref]), model.encode([context]))[0][0])

        recalls.append(recall)
        semantics.append(semantic)
        print(f"{'✅' if semantic >= 0.5 else '❌'} [{qid}] {item['query']}")
        print(f"   Recall: {recall:.0%}  |  Semantic: {semantic:.4f}")

    print(f"\nAvg Recall: {np.mean(recalls):.0%}  |  Avg Semantic: {np.mean(semantics):.4f}\n")


def evaluate_generation(results, refs, bert_model):
    from bert_score import BERTScorer
    from tqdm import tqdm

    scorer    = BERTScorer(model_type=bert_model)
    f1_scores = []

    for item in tqdm(results, desc="BERTScore"):
        qid = int(item["query_id"])
        if qid not in refs:
            continue
        with open(os.devnull, "w") as devnull:
            _out, _err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = devnull
            try:
                _, _, f1 = scorer.score([item["response"]], [refs[qid]])
            finally:
                sys.stdout, sys.stderr = _out, _err
        f1_scores.append(f1.item())

    print(f"\nAvg BERTScore F1: {np.mean(f1_scores):.4f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload",         required=True)
    parser.add_argument("--test-cases",      required=True)
    parser.add_argument("--mode",            choices=["retrieval", "generation", "both"], default="both")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--bert-model",      default="roberta-large")
    args = parser.parse_args()

    refs    = load_references(args.test_cases)
    results = json.load(open(args.payload))["results"]

    if args.mode in ("retrieval", "both"):
        evaluate_retrieval(results, refs, args.embedding_model)
    if args.mode in ("generation", "both"):
        evaluate_generation(results, refs, args.bert_model)


if __name__ == "__main__":
    main()