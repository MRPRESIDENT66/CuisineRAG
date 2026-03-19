import json

class RAGPipeline:

    def __init__(self, chunker, embedder, vectordb, retriever, prompt_builder, llm):

        self.chunker = chunker
        self.embedder = embedder
        self.vectordb = vectordb
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm


    def index_data(self, filepaths):
        """Load and index one or more JSON corpus files.

        Args:
            filepaths: a single path string or a list of path strings.
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        # 1. Load and merge all corpora
        corpus = []
        for fp in filepaths:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            corpus.extend(data)
            print(f"Loaded {len(data):,} docs from {fp}")

        if not corpus:
            raise ValueError("All corpora are empty!")

        # 2. Chunk documents — returns list[Document]
        print(f"Indexing {len(corpus):,} documents total...")
        self.chunks = self.chunker.chunk(corpus)
        for idx, chunk in enumerate(self.chunks):
            chunk.metadata['chunk_id'] = idx

        # 3. Extract text and generate embeddings
        texts = [doc.page_content for doc in self.chunks]
        embeddings = self.embedder.embed_documents(texts)

        # 4. Store Document objects + embeddings so metadata is available at retrieval time
        self.vectordb.add_documents(embeddings, self.chunks)
        print(f"Finished! Total chunks: {len(self.chunks)}")


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