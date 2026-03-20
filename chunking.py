from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


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

            # 1. Handle summary
            raw_summary = doc.get('summary') or ''
            summary = (raw_summary.get('section_text', '') if isinstance(raw_summary, dict) else raw_summary).strip()
            if summary:
                metadata = {
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