class RAGPipeline:

    def __init__(self, chunker, embedder, vectordb, retriever, prompt_builder, llm):

        self.chunker = chunker
        self.embedder = embedder
        self.vectordb = vectordb
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm


    def load_text(self, filepath):

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        return text


    def index_data(self, filepath):

        text = self.load_text(filepath)

        chunks = self.chunker.chunk(text)

        print(f"Total chunks: {len(chunks)}")

        embeddings = self.embedder.embed_documents(chunks)

        print("Adding to vector database...")

        self.vectordb.add_documents(
            embeddings,
            chunks
        )

        self.chunks = chunks   # stored so Retriever can build BM25 index


    def query(self, question):

        query_embedding = self.embedder.embed_query(question)

        retrieved_docs = self.retriever.retrieve(
            question,          # raw text — needed for BM25
            query_embedding    # vector   — needed for dense search
        )

        prompt = self.prompt_builder.build_prompt(
            question,
            retrieved_docs
        )

        answer = self.llm.generate(prompt)

        return answer, retrieved_docs