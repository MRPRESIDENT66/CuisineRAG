import importlib.metadata
import re
from typing import Any, Iterable

import numpy as np
from langchain_core.documents import Document


HEADER_PATTERN = re.compile(r"^\s*#{1,6}\s+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def split_text_into_semantic_units(text: str) -> list[str]:
    units: list[str] = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        if HEADER_PATTERN.match(stripped):
            units.append(stripped)
            continue

        sentences = [part.strip() for part in SENTENCE_PATTERN.split(stripped) if part.strip()]
        units.extend(sentences)

    return units


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0

    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0.0:
        return 0.0

    return float(np.dot(left, right) / denominator)


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def load_sentence_transformer(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:
        st_version = _package_version("sentence-transformers")
        transformers_version = _package_version("transformers")
        raise RuntimeError(
            "Failed to initialize SentenceTransformer. "
            f"Installed versions: sentence-transformers={st_version}, "
            f"transformers={transformers_version}. "
            "This usually means the notebook kernel is using stale or incompatible "
            "packages. Restart the kernel after reinstalling dependencies and make "
            "sure these two packages come from the same environment."
        ) from exc


def _chunk_text(
    text: str,
    metadata: dict,
    model: Any,
    similarity_threshold: float,
    min_chunk_size: int,
    max_chunk_size: int,
    overlap_units: int,
) -> list[Document]:
    units = split_text_into_semantic_units(text)
    if not units:
        return []

    embeddings = model.encode(units, convert_to_numpy=True, normalize_embeddings=True)

    chunks: list[Document] = []
    current_units = [units[0]]
    current_embeddings = [embeddings[0]]

    for unit, embedding in zip(units[1:], embeddings[1:]):
        current_text = " ".join(current_units)
        current_centroid = np.mean(np.vstack(current_embeddings), axis=0)
        similarity = cosine_similarity(current_centroid, embedding)
        proposed_text = f"{current_text} {unit}".strip()

        should_split = (
            len(current_text) >= min_chunk_size
            and (
                len(proposed_text) > max_chunk_size
                or similarity < similarity_threshold
            )
        )

        if should_split:
            chunks.append(Document(page_content=current_text, metadata=dict(metadata)))
            overlap_count = max(0, overlap_units)
            if overlap_count:
                current_units = current_units[-overlap_count:] + [unit]
                current_embeddings = current_embeddings[-overlap_count:] + [embedding]
            else:
                current_units = [unit]
                current_embeddings = [embedding]
            continue

        current_units.append(unit)
        current_embeddings.append(embedding)

    final_text = " ".join(current_units).strip()
    if final_text:
        chunks.append(Document(page_content=final_text, metadata=dict(metadata)))

    return chunks


def semantic_chunk_documents(
    texts: Iterable[str],
    metadatas: Iterable[dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold: float = 0.55,
    min_chunk_size: int = 180,
    max_chunk_size: int = 450,
    overlap_units: int = 1,
) -> list[Document]:
    model = load_sentence_transformer(model_name)
    chunks: list[Document] = []

    for text, metadata in zip(texts, metadatas):
        if not text or not text.strip():
            continue
        chunks.extend(
            _chunk_text(
                text=text,
                metadata=metadata,
                model=model,
                similarity_threshold=similarity_threshold,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                overlap_units=overlap_units,
            )
        )

    return chunks
