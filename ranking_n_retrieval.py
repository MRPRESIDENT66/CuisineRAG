# class CosineRetriever
# class MMRRetriever

class Retriever:

    def __init__(self, vectordb):

        self.vectordb = vectordb


    def retrieve(self, query_embedding, k=3):

        results = self.vectordb.search(
            query_embedding,
            k
        )

        return results