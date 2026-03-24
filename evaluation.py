"""
evaluate_rag.py — RAG Pipeline Evaluation

    from evaluate_rag import evaluate_retrieval, evaluate_generation

    evaluate_retrieval("output_payload.json", "test_cases.json")
    evaluate_generation("output_payload.json", "test_cases.json")

Input schemas:
    output_payload.json   {"results": [{"query_id", "query", "response", "retrieved_context": [{"doc_id", "text"}]}]}
    test_cases.json       [{"id": 1, "reference": "..."}]  ← ids are 1-indexed

Requirements:
    pip install sentence-transformers scikit-learn numpy bert-score tqdm
"""

import json, os, re, sys
import numpy as np
from typing import Dict

STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","of","and","or","but","for",
    "with","from","that","this","was","are","be","as","by","its","where","what",
    "how","did","does","do","has","have","been","not","which","who","they",
    "their","also","into","can",
}


def _load_references(test_cases_path: str) -> Dict[int, str]:
    test_cases = json.load(open(test_cases_path))
    return {tc["id"] - 1: tc["reference"] for tc in test_cases}


def evaluate_retrieval(
    payload_path: str,
    test_cases_path: str,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Dict:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    refs    = _load_references(test_cases_path)
    results = json.load(open(payload_path))["results"]
    model   = SentenceTransformer(embedding_model)
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
    return {"avg_recall": float(np.mean(recalls)), "avg_semantic": float(np.mean(semantics))}


def evaluate_generation(
    payload_path: str,
    test_cases_path: str,
    bert_model: str = "roberta-large",
) -> Dict:
    from bert_score import BERTScorer
    from tqdm import tqdm

    refs      = _load_references(test_cases_path)
    results   = json.load(open(payload_path))["results"]
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
    return {"avg_bert_f1": float(np.mean(f1_scores))}

def _extract_keywords(text: str) -> str:
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return " ".join(t for t in tokens if t not in STOPWORDS)
 
 
def evaluate_sacrebleu(
    payload_path: str,
    test_cases_path: str,
) -> Dict:
    bleu_scorer = BLEU(effective_order=True)
    chrf_scorer = CHRF(beta=3, word_order=2)
 
    refs    = _load_references(test_cases_path)
    results = json.load(open(payload_path))["results"]
    bleu_scores, chrf_scores, per_query, skipped = [], [], [], []
 
    with tqdm(results, desc="SacreBLEU + ChrF3++ evaluation", ncols=80) as pbar:
        for item in pbar:
            query_id = int(item["query_id"])
            if query_id not in refs:
                skipped.append(query_id)
                continue
 
            hypothesis = _extract_keywords(item["response"])
            reference  = _extract_keywords(refs[query_id])
 
            bleu_score = bleu_scorer.sentence_score(hypothesis, [reference]).score
            chrf_score = chrf_scorer.sentence_score(hypothesis, [reference]).score
 
            bleu_scores.append(bleu_score)
            chrf_scores.append(chrf_score)
            per_query.append({"query_id": query_id, "sacrebleu": round(bleu_score, 4), "chrf3++": round(chrf_score, 4)})
            pbar.set_postfix({"avg_BLEU": f"{np.mean(bleu_scores):.2f}", "avg_ChrF3++": f"{np.mean(chrf_scores):.2f}"})
 
    avg_bleu = float(np.mean(bleu_scores))
    avg_chrf = float(np.mean(chrf_scores))
    print(f"\nAverage SacreBLEU : {avg_bleu:.2f}")
    print(f"Average ChrF3++   : {avg_chrf:.2f}")
    if skipped:
        print(f"Skipped (no reference): {skipped}")
    return {"avg_sacrebleu": avg_bleu, "avg_chrf3++": avg_chrf, "per_query": per_query}
 