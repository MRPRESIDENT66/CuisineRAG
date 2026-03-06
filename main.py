from chunking import SimpleChunker
from embeddings import MiniLMEmbedding, QwenEmbedding
from vectore_store import FAISSVectorDB, ChromaVectorDB
from ranking_n_retrieval import Retriever
from llm_n_prompt import QwenLLM, PromptTemplate
from rag_pipeline import RAGPipeline


def main():

    filepath = "data/south_asian_cuisine.txt"

    chunker = SimpleChunker()

    # choose embedding

    embedder = MiniLMEmbedding()
    # embedder = QwenEmbedding()

    # choose vector db

    vectordb = FAISSVectorDB(dim=384)
    # vectordb = ChromaVectorDB()

    retriever = Retriever(vectordb)

    prompt_builder = PromptTemplate()

    # choose device

    llm = QwenLLM(device="cpu")
    # llm = QwenLLM(device="cuda") ##GPU support if available



    pipeline = RAGPipeline(
        chunker,
        embedder,
        vectordb,
        retriever,
        prompt_builder,
        llm
    )


    pipeline.index_data(filepath)


    question = "What spices are commonly used in South Asian cooking?"

    answer, docs = pipeline.query(question)


    print("\nRetrieved Context:\n")

    for i, d in enumerate(docs):

        print(f"Doc {i+1}:\n{d}\n")
        print("-"*40)


    print("\nFinal Answer:\n")

    print(answer)



if __name__ == "__main__":
    main()