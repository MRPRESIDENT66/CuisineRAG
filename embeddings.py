from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch


class MiniLMEmbedding:

    def __init__(self):
        print("Loading MiniLM embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings

    def embed_query(self, query):

        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        return embedding


class QwenEmbedding:

    def __init__(self):

        print("Loading Qwen embedding model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B"
        )

        self.model = AutoModel.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B"
        )

        self.model.eval()


    def mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output[0]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1),
            min=1e-9
        )


    def embed_documents(self, texts):

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = self.model(**inputs)

        embeddings = self.mean_pooling(outputs, inputs["attention_mask"])

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.numpy()


    def embed_query(self, query):

        return self.embed_documents([query])[0]