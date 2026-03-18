from chunking import SimpleChunker
from embeddings import MiniLMEmbedding, QwenEmbedding
from vectore_store import FAISSVectorDB, ChromaVectorDB
from ranking_n_retrieval import Retriever
from llm_n_prompt import QwenLLM, PromptTemplate
from rag_pipeline import RAGPipeline


# ==============================================================
# CONFIGURATION — change these to test different combinations
# ==============================================================

CHUNKER  = "simple"   # only one implemented for now
EMBEDDING  = "minilm"   # "minilm"  or  "qwen"
VECTORDB   = "faiss"    # "faiss"   or  "chroma"
RETRIEVAL  = "combo5"   # "combo4"  or  "combo5"
DEVICE     = "cpu"      # "cpu"     or  "cuda"
FILEPATH   = "data/south_asian_cuisine.txt"

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
    chunker       = SimpleChunker()
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

    print(f"\nIndexing: {FILEPATH}")
    pipeline.index_data(FILEPATH)

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
        for i, doc in enumerate(docs, 1):
            print(f"[{i}] {doc[:120]}...")
        print("-" * 40 + "\n")


if __name__ == "__main__":
    main()