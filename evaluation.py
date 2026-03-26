"""
evaluate_rag.py — Consolidated RAG Pipeline Evaluation

Usage:
    from evaluate_rag import evaluate_rag_pipeline
    evaluate_rag_pipeline(output_path, benchmark_path)

Input schemas:
    output_path   (JSON) : {"results": [{"query_id", "query", "response", "retrieved_context": [{"doc_id", "text"}]}]}
    benchmark_path (JSON): [{"id": 1, "query": "...", "reference": "...", "relevant_doc_ids": [...]}]

Requirements:
    pip install sentence-transformers scikit-learn numpy bert-score sacrebleu tqdm nltk pandas
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","of","and","or","but","for",
    "with","from","that","this","was","are","be","as","by","its","where","what",
    "how","did","does","do","has","have","been","not","which","who","they",
    "their","also","into","can",
}

def _extract_keywords(text: str) -> str:
    """Strip stopwords and return remaining tokens as a joined string."""
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return " ".join(t for t in tokens if t not in STOPWORDS)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_rag_pipeline(
    output_path: str,
    benchmark_path: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    bert_model: str = "roberta-large",
) -> pd.DataFrame:
    """
    Runs all evaluations on the RAG pipeline outputs and prints a full report.

    Metrics covered
    ---------------
    Generation : BLEU (nltk), ROUGE-1 (F1 / Precision / Recall),
                 SacreBLEU, ChrF3++, BERTScore F1
    Retrieval  : Keyword Recall, Semantic Similarity,
                 Precision, Recall, MRR
    """

    # ── 0. LAZY IMPORTS (so missing packages fail loudly at call-time) ────────
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from bert_score import BERTScorer
    from sacrebleu.metrics import BLEU as SacreBLEU, CHRF

    # ── 1. LOAD DATA ──────────────────────────────────────────────────────────
    print("=" * 65)
    print("  RAG PIPELINE EVALUATION")
    print("=" * 65)

    with open(output_path, "r") as f:
        output_data = json.load(f)
    with open(benchmark_path, "r") as f:
        benchmark_data = json.load(f)

    results = output_data["results"]

    # Build lookup maps — support both {id + reference} and {query + reference} schemas
    refs_by_id    = {item["id"] - 1: item["reference"] for item in benchmark_data if "id" in item}
    refs_by_query = {item["query"]: item["reference"]  for item in benchmark_data if "query" in item}
    docs_by_query = {item["query"]: set(item["relevant_doc_ids"])
                     for item in benchmark_data if "relevant_doc_ids" in item}

    def _get_ref(item):
        """Return (reference_text, true_doc_ids) for a result item, or (None, None)."""
        ref = refs_by_id.get(int(item["query_id"])) or refs_by_query.get(item["query"])
        true_docs = docs_by_query.get(item.get("query", ""), set())
        return ref, true_docs

    # ── 2. INITIALISE MODELS & SCORERS ────────────────────────────────────────
    print("\n[Setup] Loading models …")
    embed_model   = SentenceTransformer(embedding_model)
    bert_scorer   = BERTScorer(model_type=bert_model)
    sacre_bleu    = SacreBLEU(effective_order=True)
    chrf_scorer   = CHRF(beta=3, word_order=2)
    nltk_smoother = SmoothingFunction().method1
    total_number_of_chunks=9888
    #12616 - for semantic
    print("[Setup] Done.\n")

    # ── 3. PER-QUERY EVALUATION ───────────────────────────────────────────────
    evaluation_results = []
    skipped = []

    for item in tqdm(results, desc="Evaluating queries", ncols=80):
        ref, true_docs = _get_ref(item)
        if ref is None:
            skipped.append(item.get("query_id"))
            continue

        query         = item["query"]
        response      = item["response"]
        context_text  = " ".join(c["text"].lower() for c in item["retrieved_context"])
        # ── GENERATION: BLEU (nltk) ──────────────────────────────────────────
        ref_tokens  = ref.lower().split()
        resp_tokens = response.lower().split()
        bleu_nltk   = sentence_bleu([ref_tokens], resp_tokens, smoothing_function=nltk_smoother)

        # ── GENERATION: ROUGE-1 ──────────────────────────────────────────────
        ref_set   = set(ref_tokens)
        resp_set  = set(resp_tokens)
        overlap   = ref_set & resp_set
        r1_recall    = len(overlap) / len(ref_set)  if ref_set  else 0.0
        r1_precision = len(overlap) / len(resp_set) if resp_set else 0.0
        r1_f1 = (
            2 * r1_precision * r1_recall / (r1_precision + r1_recall)
            if (r1_precision + r1_recall) > 0 else 0.0
        )

        # ── GENERATION: SacreBLEU + ChrF3++ ──────────────────────────────────
        hyp_kw  = _extract_keywords(response)
        ref_kw  = _extract_keywords(ref)
        sacre_b = sacre_bleu.sentence_score(hyp_kw, [ref_kw]).score
        chrf    = chrf_scorer.sentence_score(hyp_kw, [ref_kw]).score
        # ── GENERATION: BERTScore ─────────────────────────────────────────────
        with open(os.devnull, "w") as devnull:
            _out, _err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = devnull
            try:
                _, _, bert_f1 = bert_scorer.score([response], [ref])
            finally:
                sys.stdout, sys.stderr = _out, _err
        bert_f1_val = float(bert_f1.item())

        # ── RETRIEVAL: Keyword Recall ─────────────────────────────────────────
        keywords  = [t for t in re.findall(r"\b[a-zA-Z]{3,}\b", ref.lower()) if t not in STOPWORDS]
        hits_kw   = [kw for kw in keywords if kw in context_text]
        kw_recall = len(hits_kw) / len(keywords) if keywords else 0.0
        # ── RETRIEVAL: Semantic Similarity ────────────────────────────────────
        semantic_sim = (cosine_similarity(embed_model.encode([ref]), embed_model.encode([c["text"].lower() for c in item["retrieved_context"]])))
        # print(semantic_sim)
        #float(
         #   cosine_similarity(embed_model.encode([ref]), embed_model.encode([c["text"].lower() for c in item["retrieved_context"]]))
        #)
        hit_doc_count=(sum(float(sim)>0.6 for sim in semantic_sim[0]))
        # print("h d c -",hit_doc_count)
        # ── RETRIEVAL: Precision, Recall, MRR ────────────────────────────────
        ret_prec   = hit_doc_count / 5
        ret_recall = hit_doc_count / total_number_of_chunks
        mrr=1/int(np.argmax(semantic_sim)+1)
        context_text  = " ".join(c["text"].lower() for c in item["retrieved_context"])
        semantic_sim = float(
            cosine_similarity(embed_model.encode([ref]), embed_model.encode([context_text]))[0][0]
        )

        evaluation_results.append({
            "query":             query,
            # Generation
            "bleu_nltk":         bleu_nltk,
            "rouge1_f1":         r1_f1,
            "rouge1_precision":  r1_precision,
            "rouge1_recall":     r1_recall,
            "sacrebleu":         sacre_b,
            "chrf3++":           chrf,
            "bert_f1":           bert_f1_val,
            # Retrieval
            "kw_recall":         kw_recall,
            "semantic_sim":      semantic_sim,
            "ret_precision":     ret_prec,
            "ret_recall":        ret_recall,
            "mrr":               mrr,
        })
        # return
    # ── 4. AGGREGATE & PRINT RESULTS ─────────────────────────────────────────
    df = pd.DataFrame(evaluation_results)

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)

    # ── GENERATION ────────────────────────────────────────────────────────────
    print("\n── GENERATION METRICS ──────────────────────────────────────")
    print(f"  BLEU (nltk)          : {df['bleu_nltk'].mean():.4f}")
    print(f"  ROUGE-1 F1           : {df['rouge1_f1'].mean():.4f}")
    print(f"  ROUGE-1 Precision    : {df['rouge1_precision'].mean():.4f}")
    print(f"  ROUGE-1 Recall       : {df['rouge1_recall'].mean():.4f}")
    print(f"  SacreBLEU            : {df['sacrebleu'].mean():.2f}")
    print(f"  ChrF3++              : {df['chrf3++'].mean():.2f}")
    print(f"  BERTScore F1         : {df['bert_f1'].mean():.4f}")

    # ── RETRIEVAL ─────────────────────────────────────────────────────────────
    print("\n── RETRIEVAL METRICS ───────────────────────────────────────")
    print(f"  Keyword Recall       : {df['kw_recall'].mean():.2%}")
    print(f"  Semantic Similarity  : {df['semantic_sim'].mean():.4f}")
    print(f"  Precision            : {df['ret_precision'].mean():.4f}")
    print(f"  Recall               : {df['ret_recall'].mean():.4f}")
    print(f"  MRR                  : {df['mrr'].mean():.4f}")

    if skipped:
        print(f"\n  ⚠  Skipped (no reference found): {skipped}")

    print("\n" + "=" * 65)

    return df


if __name__=='__main__':
    output_path="data/output_payload_sample_benchmark.json"
    benchmark_path="data/benchmark_dataset.json"
    evaluate_rag_pipeline(output_path,benchmark_path)