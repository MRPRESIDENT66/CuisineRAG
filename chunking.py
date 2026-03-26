import re

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


class SectionAwareChunker:
    def __init__(self, chunk_size=800, chunk_overlap=120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Recursive splitter config mirroring the notebook setup
        self.section_recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Sections to skip (low information value)
        self.ignored_sections = {
            'see also', 'references', 'external links',
            'further reading', 'notes'
        }

    def chunk(self, corpus):
        """
        Input:  corpus list loaded from JSON
        Output: list[Document]
        """
        section_based_chunks = []

        for doc in corpus:
            title = doc.get('title', '')
            url = doc.get('url', '')
            doc_id = doc.get('doc_id', '')

            # 1. Handle summary
            raw_summary = doc.get('summary') or ''
            summary = (raw_summary.get('section_text', '') if isinstance(raw_summary, dict) else raw_summary).strip()
            if summary:
                metadata = {
                    'doc_id': doc_id,
                    'title': title,
                    'url': url,
                    'section': 'Summary',
                }
                section_based_chunks.extend(self._split_and_meta(summary, metadata))

            # 2. Handle sections
            for sec in doc.get('content', []):
                sec_title = (sec.get('section_title') or '').strip()
                sec_text = (sec.get('section_text') or '').strip()

                if not sec_text or sec_text == summary:
                    continue

                # Parse nested titles (e.g. "History / Ingredients")
                section_name = sec_title
                subsection_name = None
                if ' / ' in sec_title:
                    section_name, subsection_name = [part.strip() for part in sec_title.split(' / ', 1)]

                # Skip ignored sections
                if section_name.lower() in self.ignored_sections:
                    continue

                metadata = {
                    'doc_id': doc_id,
                    'title': title,
                    'url': url,
                    'section': section_name,
                }
                if subsection_name:
                    metadata['subsection'] = subsection_name

                section_based_chunks.extend(self._split_and_meta(sec_text, metadata))

        return section_based_chunks

    def _split_and_meta(self, text, base_metadata):
        """Split text and inject subchunk_id into metadata."""
        # Use create_documents to keep metadata associated with each chunk
        docs = self.section_recursive_splitter.create_documents([text], metadatas=[base_metadata])

        for chunk in docs:
            chunk.page_content = chunk.page_content.strip()

        return docs


# ──────────────────────────────────────────────────────────
#  Semantic Chunker (embedding-based topic-shift detection)
# ──────────────────────────────────────────────────────────

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker:
    """
    Section-aware semantic chunker: respects JSON section boundaries,
    uses embedding-based topic-shift detection *within* each section
    instead of RecursiveCharacterTextSplitter.
    Returns list[Document] — same interface as SectionAwareChunker.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.50,
        min_chunk_size: int = 240,
        max_chunk_size: int = 800,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Sections to skip (same as SectionAwareChunker)
        self.ignored_sections = {
            'see also', 'references', 'external links',
            'further reading', 'notes'
        }

        print(f"Loading embedding model for semantic chunking ({model_name})...")
        self.model = SentenceTransformer(model_name)

    # ── public API (same signature as SectionAwareChunker) ──

    def chunk(self, corpus) -> list[Document]:
        """
        Input:  corpus list loaded from JSON
        Output: list[Document]
        """
        all_chunks: list[Document] = []

        for doc in corpus:
            title = doc.get('title', '')
            url = doc.get('url', '')
            doc_id = doc.get('doc_id', '')

            # 1. Handle summary
            raw_summary = doc.get('summary') or ''
            summary = (raw_summary.get('section_text', '') if isinstance(raw_summary, dict) else raw_summary).strip()
            if summary:
                metadata = {
                    'doc_id': doc_id,
                    'title': title,
                    'url': url,
                    'section': 'Summary',
                }
                all_chunks.extend(self._split_section(summary, metadata))

            # 2. Handle sections
            for sec in doc.get('content', []):
                sec_title = (sec.get('section_title') or '').strip()
                sec_text = (sec.get('section_text') or '').strip()

                if not sec_text or sec_text == summary:
                    continue

                # Parse nested titles
                section_name = sec_title
                subsection_name = None
                if ' / ' in sec_title:
                    section_name, subsection_name = [part.strip() for part in sec_title.split(' / ', 1)]

                if section_name.lower() in self.ignored_sections:
                    continue

                metadata = {
                    'doc_id': doc_id,
                    'title': title,
                    'url': url,
                    'section': section_name,
                }
                if subsection_name:
                    metadata['subsection'] = subsection_name

                all_chunks.extend(self._split_section(sec_text, metadata))

        return all_chunks

    def _split_section(self, text: str, metadata: dict) -> list[Document]:
        """If section is short enough, return as-is. Otherwise, semantic split."""
        if len(text) <= self.max_chunk_size:
            return [Document(page_content=text.strip(), metadata=dict(metadata))]
        return self._semantic_split(text, metadata)

    # ── internals ──

    @staticmethod
    def _to_units(text: str) -> list[str]:
        """Split text into sentence-level units."""
        units: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            sentences = [s.strip() for s in _SENTENCE_RE.split(stripped) if s.strip()]
            units.extend(sentences)
        return units

    def _semantic_split(self, text: str, metadata: dict) -> list[Document]:
        units = self._to_units(text)
        if not units:
            return []

        embeddings = self.model.encode(units, convert_to_numpy=True, normalize_embeddings=True)

        chunks: list[Document] = []
        cur_units = [units[0]]
        centroid_sum = embeddings[0].copy()
        centroid_count = 1

        for unit, emb in zip(units[1:], embeddings[1:]):
            cur_text = " ".join(cur_units)
            centroid = centroid_sum / centroid_count
            sim = float(np.dot(centroid, emb))  # already L2-normalised

            proposed_len = len(cur_text) + 1 + len(unit)
            should_split = (
                len(cur_text) >= self.min_chunk_size
                and (proposed_len > self.max_chunk_size or sim < self.similarity_threshold)
            )

            if should_split:
                chunks.append(Document(page_content=cur_text, metadata=dict(metadata)))
                cur_units = [unit]
                centroid_sum = emb.copy()
                centroid_count = 1
            else:
                cur_units.append(unit)
                centroid_sum += emb
                centroid_count += 1

        final = " ".join(cur_units).strip()
        if final:
            chunks.append(Document(page_content=final, metadata=dict(metadata)))

        return chunks