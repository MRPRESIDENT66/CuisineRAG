# CuisineRAG 🍛

### A Modular Retrieval-Augmented Generation (RAG) System for South Asian Cuisine Knowledge

CuisineRAG is a modular **Retrieval-Augmented Generation (RAG)** system designed to answer questions about **South Asian cuisine** using a curated knowledge base. The system retrieves relevant contextual information from a dataset and uses a **Large Language Model (LLM)** to generate grounded responses.

The project is structured to allow experimentation with different **chunking methods, embedding models, vector databases, and retrieval strategies**, making it suitable for research, coursework, and educational exploration of RAG architectures.

---

# Project Overview

The system processes a **text-based knowledge base** about South Asian cuisine and enables users to ask natural language questions about dishes, ingredients, cooking techniques, spices, and culinary traditions.

The pipeline consists of several stages:

1. Text ingestion
2. Chunking of documents
3. Embedding generation
4. Vector database indexing
5. Retrieval of relevant chunks
6. Prompt construction
7. Answer generation using an LLM

---

# RAG Pipeline Architecture

```
Raw Text Knowledge Base
        │
        ▼
Document Chunking
        │
        ▼
Embedding Generation
        │
        ▼
Vector Database Indexing
        │
        ▼
Query Embedding
        │
        ▼
Similarity Search (Top-K Retrieval)
        │
        ▼
Prompt Construction
        │
        ▼
LLM Generation
        │
        ▼
Final Answer
```

This architecture ensures the language model **uses retrieved context rather than relying only on its internal knowledge**, reducing hallucinations and improving factual grounding.

---

# Project Structure

```
CuisineRAG/
│
├── data/
│   └── south_asian_cuisine.txt
│
├── chunking.py
├── embeddings.py
├── vector_store.py
├── ranking_n_retrieval.py
├── llm_n_prompt.py
├── rag_pipeline.py
│
├── evaluation.py
├── main.py
│
├── notebooks
│   ├── chunking.ipynb
│   ├── pipeline.ipynb
│   ├── scraping.ipynb
│   └── evaluation.ipynb
│
├── requirements.txt
└── README.md
```

---

# Module Descriptions

## `chunking.py`

Responsible for splitting the input text into smaller manageable chunks.

Purpose:

* Improves retrieval precision
* Ensures embeddings capture meaningful semantic segments

Typical functionality:

* Fixed-size chunking
* Optional overlap between chunks

Output:

```
Text Document → List of Text Chunks
```

---

# `embeddings.py`

Generates vector representations of text chunks using embedding models.

Supported models:

| Model          | Type                  | Dimensions |
| -------------- | --------------------- | ---------- |
| MiniLM         | Sentence Transformers | 384        |
| Qwen Embedding | Transformer-based     | 1024       |

Responsibilities:

* Convert text chunks into vector embeddings
* Generate query embeddings during inference

Output:

```
Text → Numerical Embedding Vector
```

---

# `vector_store.py`

Handles storage and similarity search for embeddings.

Supported vector databases:

| Database | Type       | Use Case                |
| -------- | ---------- | ----------------------- |
| FAISS    | In-memory  | Fast similarity search  |
| ChromaDB | Persistent | Scalable vector storage |

Responsibilities:

* Store document embeddings
* Perform similarity search
* Return top-K relevant chunks

---

# `ranking_n_retrieval.py`

Implements retrieval logic.

Responsibilities:

* Query embedding generation
* Similarity search
* Returning top-K relevant context chunks

Output:

```
User Query → Relevant Context Chunks
```

---

# `llm_n_prompt.py`

Handles prompt construction and language model inference.

LLM Used:

```
Qwen/Qwen2.5-0.5B-Instruct
```

Capabilities:

* Context-aware answer generation
* Instruction-following responses
* CPU or GPU inference support

Prompt structure:

```
System Instructions
+
Retrieved Context
+
User Question
```

---

# `rag_pipeline.py`

The central orchestration module connecting all components.

Responsibilities:

1. Load dataset
2. Chunk documents
3. Generate embeddings
4. Store embeddings in vector database
5. Retrieve relevant documents
6. Construct prompt
7. Generate final answer

This module ensures all RAG components work together seamlessly.

---

# `main.py`

Entry point for running the full pipeline.

Typical workflow:

```
Load dataset
↓
Build vector index
↓
Ask user query
↓
Retrieve relevant chunks
↓
Generate answer
↓
Display results
```

---

# Evaluation Module

`evaluation.py` enables testing and benchmarking of different configurations.

Possible experiments include:

* Comparing embedding models
* Testing vector databases
* Evaluating chunking strategies
* Measuring retrieval accuracy

Metrics that can be evaluated:

* Retrieval relevance
* Context overlap
* Response quality

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/CuisineRAG.git
cd CuisineRAG
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Dependencies

Key libraries used:

```
sentence-transformers
transformers
torch
faiss-cpu
chromadb
numpy
```

---

# Running the System

Run the full pipeline:

```
python main.py
```

Example query:

```
What spices are commonly used in South Asian cooking?
```

Example output:

```
Common spices used in South Asian cooking include:

• Turmeric
• Cumin
• Coriander
• Cardamom
• Fenugreek
• Mustard seeds
• Garam masala

These spices provide the distinctive aroma and flavor that define many South Asian dishes.
```

---

# Example Use Cases

CuisineRAG can answer questions such as:

* What spices are used in biryani?
* What is dal makhani?
* How is naan traditionally cooked?
* What are common South Asian cooking techniques?
* What drinks are popular in South Asian cuisine?

---

# Key Design Goals

The project focuses on:

* **Modularity** — each component can be swapped easily
* **Reproducibility** — experiments can be repeated
* **Transparency** — clear RAG pipeline
* **Educational value** — useful for learning RAG systems

---

# Acknowledgements

This project builds upon open-source tools from:

* HuggingFace Transformers
* SentenceTransformers
* FAISS
* ChromaDB

