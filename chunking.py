import re
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
class SectionAwareChunker:
    def __init__(self, chunk_size=800, chunk_overlap=120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.section_recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.ignored_sections = {
            'see also', 'references', 'external links',
            'further reading', 'notes'
        }

    def chunk(self, corpus):
        section_based_chunks = []
        global_chunk_idx = 1

        for doc in corpus:
            title = doc.get('title', '')
            url = doc.get('url', '')

            # 1. Handle summary
            raw_summary = doc.get('summary') or ''
            summary = (raw_summary.get('section_text', '') if isinstance(raw_summary, dict) else raw_summary).strip()
            if summary:
                metadata = {
                    'title': title,
                    'url': url,
                    'section': 'Summary',
                }
                docs, global_chunk_idx = self._split_and_meta(summary, metadata, global_chunk_idx)
                section_based_chunks.extend(docs)

            # 2. Handle sections
            for sec in doc.get('content', []):
                sec_title = (sec.get('section_title') or '').strip()
                sec_text = (sec.get('section_text') or '').strip()

                if not sec_text or sec_text == summary:
                    continue

                section_name = sec_title
                subsection_name = None
                if ' / ' in sec_title:
                    section_name, subsection_name = [part.strip() for part in sec_title.split(' / ', 1)]

                if section_name.lower() in self.ignored_sections:
                    continue

                metadata = {
                    'title': title,
                    'url': url,
                    'section': section_name,
                }
                if subsection_name:
                    metadata['subsection'] = subsection_name

                docs, global_chunk_idx = self._split_and_meta(sec_text, metadata, global_chunk_idx)
                section_based_chunks.extend(docs)

        return section_based_chunks

    def _split_and_meta(self, text, base_metadata,current_idx):
        """Split text and inject a global numeric chunk_id into metadata."""
        docs = self.section_recursive_splitter.create_documents([text], metadatas=[base_metadata])

        for chunk in docs:
            chunk.page_content = chunk.page_content.strip()
            chunk.metadata['chunk_id'] = current_idx
            current_idx += 1

        return docs,current_idx


class SemanticChunker:
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
        self.ignored_sections = {
            'see also', 'references', 'external links',
            'further reading', 'notes'
        }

        print(f"Loading embedding model for semantic chunking ({model_name})...")
        self.model = SentenceTransformer(model_name)

    def chunk(self, corpus) -> list[Document]:
        all_chunks: list[Document] = []
        global_chunk_idx = 1

        for doc in corpus:
            title = doc.get('title', 'UnknownDoc')
            url = doc.get('url', '')

            # 1. Summary
            raw_summary = doc.get('summary') or ''
            summary = (raw_summary.get('section_text', '') if isinstance(raw_summary, dict) else raw_summary).strip()
            if summary:
                metadata = {'title': title, 'url': url, 'section': 'Summary'}
                docs, global_chunk_idx = self._split_section(summary, metadata, global_chunk_idx)
                all_chunks.extend(docs)

            # 2. Content Sections
            for sec in doc.get('content', []):
                sec_title = (sec.get('section_title') or '').strip()
                sec_text = (sec.get('section_text') or '').strip()

                if not sec_text or sec_text == summary:
                    continue

                section_name = sec_title
                subsection_name = None
                if ' / ' in sec_title:
                    parts = [part.strip() for part in sec_title.split(' / ', 1)]
                    section_name = parts[0]
                    subsection_name = parts[1] if len(parts) > 1 else None

                if section_name.lower() in self.ignored_sections:
                    continue

                metadata = {'title': title, 'url': url, 'section': section_name}
                if subsection_name:
                    metadata['subsection'] = subsection_name

                docs, global_chunk_idx = self._split_section(sec_text, metadata, global_chunk_idx)
                all_chunks.extend(docs)

        return all_chunks

    def _split_section(self, text: str, metadata: dict, current_idx: int):
        if len(text) <= self.max_chunk_size:
            meta = dict(metadata)
            meta['chunk_id'] = current_idx
            current_idx += 1
            return [Document(page_content=text.strip(), metadata=meta)], current_idx

        return self._semantic_split(text, metadata, current_idx)

    @staticmethod
    def _to_units(text: str) -> list[str]:
        units: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped: continue
            sentences = [s.strip() for s in _SENTENCE_RE.split(stripped) if s.strip()]
            units.extend(sentences)
        return units

    def _semantic_split(self, text: str, metadata: dict, current_idx: int):
        units = self._to_units(text)
        if not units: return [], current_idx

        embeddings = self.model.encode(units, convert_to_numpy=True, normalize_embeddings=True)

        chunks: list[Document] = []
        cur_units = [units[0]]
        centroid_sum = embeddings[0].copy()
        centroid_count = 1

        for unit, emb in zip(units[1:], embeddings[1:]):
            cur_text = " ".join(cur_units)
            centroid = centroid_sum / centroid_count
            sim = float(np.dot(centroid, emb))

            proposed_len = len(cur_text) + 1 + len(unit)
            should_split = (
                    len(cur_text) >= self.min_chunk_size
                    and (proposed_len > self.max_chunk_size or sim < self.similarity_threshold)
            )

            if should_split:
                meta_with_id = dict(metadata)
                meta_with_id['chunk_id'] = current_idx
                chunks.append(Document(page_content=cur_text, metadata=meta_with_id))

                current_idx += 1
                cur_units = [unit]
                centroid_sum = emb.copy()
                centroid_count = 1
            else:
                cur_units.append(unit)
                centroid_sum += emb
                centroid_count += 1

        final = " ".join(cur_units).strip()
        if final:
            meta_with_id = dict(metadata)
            meta_with_id['chunk_id'] = current_idx
            chunks.append(Document(page_content=final, metadata=meta_with_id))
            current_idx += 1

        return chunks, current_idx
