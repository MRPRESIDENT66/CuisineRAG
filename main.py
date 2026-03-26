import torch
import json
from chunking import SectionAwareChunker, SemanticChunker
from embeddings import MiniLMEmbedding, QwenEmbedding
from vectore_store import FAISSVectorDB
from ranking_n_retrieval import Retriever
from llm_n_prompt import QwenLLM, PromptTemplate
from rag_pipeline import RAGPipeline


# ==============================================================
# CONFIGURATION — change these to test different combinations
# ==============================================================
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

CHUNKER    = "section"  # "section" or "semantic"
EMBEDDING  = "minilm"   # "minilm"  or  "qwen"
VECTORDB   = "faiss"    # "faiss"   or  "chroma"
RETRIEVAL  = "combo2"   # "combo1"  or  "combo2"
DEVICE     = get_device()
FILEPATHS  = [
    "data/corpus/wikipedia_south_asian.json",
    "data/corpus/wikibook_cookbook.json",
    "data/corpus/blog_80cuisines.json"
]

# ==============================================================


def build_embedder(choice):
    if choice == "minilm":
        return MiniLMEmbedding(), 384
    elif choice == "qwen":
        return QwenEmbedding(), 1024
    else:
        raise ValueError(f"Unknown embedder: {choice}. Choose 'minilm' or 'qwen'")


def build_vectordb(choice, dim):
    if choice == "faiss":
        return FAISSVectorDB(dim=dim)
    # elif choice == "chroma":
    #     return ChromaVectorDB()
    else:
        raise ValueError(f"Unknown vectordb: {choice}. Choose 'faiss' or 'chroma'")


def build_retriever(choice, vectordb, documents):
    retriever = Retriever(vectordb, documents)
    retriever.active_combo = choice   # store choice on the object
    return retriever


def run_json_input_output():

    print("\n" + "="*50)
    print("       CuisineRAG — ChefBot")
    print("="*50)
    print(f"  Chunker: {CHUNKER}")
    print(f"  Embedding : {EMBEDDING}")
    print(f"  VectorDB  : {VECTORDB}")
    print(f"  Retrieval : {RETRIEVAL}")
    print(f"  Device    : {DEVICE}")
    print("="*50 + "\n")

    # --- build components based on config ---
    if CHUNKER == "semantic":
        chunker = SemanticChunker()       # reuses all-MiniLM-L6-v2
    else:
        chunker = SectionAwareChunker()
    embedder, dim = build_embedder(EMBEDDING)
    vectordb      = build_vectordb(VECTORDB, dim)
    prompt_builder = PromptTemplate()
    llm           = QwenLLM(device=DEVICE)

    # --- build pipeline (retriever added after indexing) ---

    pipeline = RAGPipeline(
        chunker        = chunker,
        embedder       = embedder,
        vectordb       = vectordb,
        retriever      = None,
        prompt_builder = prompt_builder,
        llm            = llm
    )

    # --- index the cuisine data ---

    import os

    INDEX_BIN  = "faiss_index.bin"
    DOCS_JSON  = "faiss_docs.json"

    if os.path.exists(INDEX_BIN) and os.path.exists(DOCS_JSON):
        # already indexed — just load from disk
        vectordb.load(INDEX_BIN, DOCS_JSON)
        pipeline.chunks = vectordb.documents
        print("Loaded existing index from disk.")
    else:
        # first run — index and save
        pipeline.index_data(FILEPATHS)
        vectordb.save(INDEX_BIN, DOCS_JSON)

    # --- build retriever with chunks for BM25 ---

    retriever = build_retriever(RETRIEVAL, vectordb, pipeline.chunks)
    pipeline.retriever = retriever

    input_file_name="data/input_payload_sample_benchmark.json"
    output_file_name= "data/output_payload_sample_benchmark.json"
    with open(input_file_name) as input_file:
        query_list=json.load(input_file)['queries']

    print(f"\nProcessing {len(query_list)} queries from {input_file_name}...\n")

    results=[0]*len(query_list)
    for i,query in enumerate(query_list):
        query_id=query["query_id"]
        query_text=query["query"].strip()
        print(f"[{i+1}/{len(query_list)}] {query_text[:60]}...")
        answer,docs=pipeline.query(query_text)
        results[i]={
            "query_id":str(query_id),
            "query":str(query_text),
            "response":str(answer),
            "retrieved_context":[{"doc_id":chunk.metadata.get('doc_id', '?'),"text":chunk.page_content} for chunk in docs]
            }
        final=dict({"results": results})
    with open(output_file_name,'w') as output_json:
        json.dump(final, output_json, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(results)} results saved to {output_file_name}")



# def main():
#
#     print("\n" + "="*50)
#     print("       CuisineRAG — ChefBot")
#     print("="*50)
#     print(f"  Chunker   : {CHUNKER}")
#     print(f"  Embedding : {EMBEDDING}")
#     print(f"  VectorDB  : {VECTORDB}")
#     print(f"  Retrieval : {RETRIEVAL}")
#     print(f"  Device    : {DEVICE}")
#     print("="*50 + "\n")
#
#     # --- build components based on config ---
#
#     embedder, dim = build_embedder(EMBEDDING)
#     vectordb      = build_vectordb(VECTORDB, dim)
#     prompt_builder = PromptTemplate()
#     llm           = QwenLLM(device=DEVICE)
#
#     # --- build pipeline (retriever added after indexing) ---
#
#     pipeline = RAGPipeline(
#         chunker        = chunker,
#         embedder       = embedder,
#         vectordb       = vectordb,
#         retriever      = None,
#         prompt_builder = prompt_builder,
#         llm            = llm
#     )
#
#     # --- index the cuisine data ---
#
#     pipeline.index_data(FILEPATHS)
#
#     # --- build retriever with chunks for BM25 ---
#
#     retriever = build_retriever(RETRIEVAL, vectordb, pipeline.chunks)
#     pipeline.retriever = retriever
#
#     print("\nReady! Type your question or 'quit' to exit.\n")
#
#     # --- chatbot loop ---
#
#     while True:
#
#         question = input("You: ").strip()
#
#         if not question:
#             continue
#
#         if question.lower() in ("quit", "exit", "q"):
#             print("Goodbye!")
#             break
#
#         print("\nChefBot: thinking...\n")
#
#         answer, docs = pipeline.query(question)
#
#         print(f"ChefBot: {answer}")
#
#         print("\n--- Retrieved chunks ---")
#         for doc in docs:
#             doc_id = doc.metadata.get('doc_id', '?')
#             chunk_id = doc.metadata.get('chunk_id', '?')
#             title = doc.metadata.get('title', '')
#             print(f"[doc={doc_id} | {chunk_id}] ({title}) {doc.page_content[:120]}...")
#         print("-" * 40 + "\n")


if __name__ == "__main__":
    run_json_input_output()