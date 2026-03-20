import faiss
import numpy as np
import chromadb


class FAISSVectorDB:

    def __init__(self, dim):

        print("Initializing FAISS...")

        self.index = faiss.IndexFlatIP(dim)
        self.documents = []


    def add_documents(self, embeddings, documents):

        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)

        self.documents.extend(documents)


    def search(self, query_embedding, k=5):

        query_embedding = np.array([query_embedding]).astype("float32")

        scores, indices = self.index.search(query_embedding, k)

        results = [self.documents[i] for i in indices[0]]

        return results
    
    def save(self, index_path="faiss_index.bin", docs_path="faiss_docs.json"):
        import json
        faiss.write_index(self.index, index_path)
        with open(docs_path, "w") as f:
            json.dump(
                [{"page_content": d.page_content, "metadata": d.metadata}
                for d in self.documents], f
            )
        print(f"FAISS saved → {index_path}, {docs_path}")

    def load(self, index_path="faiss_index.bin", docs_path="faiss_docs.json"):
        import json
        from langchain_core.documents import Document
        self.index = faiss.read_index(index_path)
        with open(docs_path, "r") as f:
            raw = json.load(f)
        self.documents = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in raw]
        print(f"FAISS loaded ← {index_path} ({len(self.documents):,} chunks)")



class ChromaVectorDB:

    def __init__(self, collection_name="rag_collection"):

        print("Initializing ChromaDB...")

        self.client = chromadb.Client()

        self.collection = self.client.create_collection(
            name=collection_name
        )


    def add_documents(self, embeddings, documents):
        ids       = [str(doc.metadata["chunk_id"]) for doc in documents]
        texts     = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )


    def search(self, query_embedding, k=5):
        from langchain_core.documents import Document

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas"]
        )

        return [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(results["documents"][0], results["metadatas"][0])
        ]