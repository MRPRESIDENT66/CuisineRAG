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



class ChromaVectorDB:

    def __init__(self, collection_name="rag_collection"):

        print("Initializing ChromaDB...")

        self.client = chromadb.Client()

        self.collection = self.client.create_collection(
            name=collection_name
        )


    def add_documents(self, embeddings, documents):

        ids = [str(i) for i in range(len(documents))]

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=ids
        )


    def search(self, query_embedding, k=5):

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )

        return results["documents"][0]