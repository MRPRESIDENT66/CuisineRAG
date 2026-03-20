import torch
from chunking import SectionAwareChunker
from embeddings import MiniLMEmbedding, QwenEmbedding
from vectore_store import FAISSVectorDB, ChromaVectorDB
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

CHUNKER  = "simple"   # only one implemented for now
EMBEDDING  = "minilm"   # "minilm"  or  "qwen"
VECTORDB   = "faiss"    # "faiss"   or  "chroma"
RETRIEVAL  = "combo2"   # "combo1"  or  "combo2"
DEVICE     = get_device()
FILEPATHS  = [
    "data/raw/south_asian_corpus.json",
    "data/raw/saved_wikibook_data.json",
    "data/raw/cuisines80.json"
]

# ==============================================================


def build_embedder(choice):
    if choice == "minilm":
        return MiniLMEmbedding(), 384
    elif choice == "qwen":
        return QwenEmbedding(), 768
    else:
        raise ValueError(f"Unknown embedder: {choice}. Choose 'minilm' or 'qwen'")


def build_vectordb(choice, dim):
    if choice == "faiss":
        return FAISSVectorDB(dim=dim)
    elif choice == "chroma":
        return ChromaVectorDB()
    else:
        raise ValueError(f"Unknown vectordb: {choice}. Choose 'faiss' or 'chroma'")


def build_retriever(choice, vectordb, documents):
    retriever = Retriever(vectordb, documents)
    retriever.active_combo = choice   # store choice on the object
    return retriever


def main():

    print("\n" + "="*50)
    print("       CuisineRAG — ChefBot")
    print("="*50)
    print(f"Chunker: {CHUNKER}")
    print(f"  Embedding : {EMBEDDING}")
    print(f"  VectorDB  : {VECTORDB}")
    print(f"  Retrieval : {RETRIEVAL}")
    print(f"  Device    : {DEVICE}")
    print("="*50 + "\n")

    # --- build components based on config ---
    chunker       = SectionAwareChunker()
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

    pipeline.index_data(FILEPATHS)

    # --- build retriever with chunks for BM25 ---

    retriever = build_retriever(RETRIEVAL, vectordb, pipeline.chunks)
    pipeline.retriever = retriever

    print("\nReady! Type your question or 'quit' to exit.\n")

    # --- chatbot loop ---

    while True:

        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nChefBot: thinking...\n")

        answer, docs = pipeline.query(question)

        print(f"ChefBot: {answer}")

        print("\n--- Retrieved chunks ---")
        for doc in docs:
            chunk_id = doc.metadata.get('chunk_id', '?')
            print(f"[{chunk_id}] {doc.page_content[:120]}...")
        print("-" * 40 + "\n")


if __name__ == "__main__":
    main()