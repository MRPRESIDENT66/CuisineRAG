# class FixedChunker
# class RecursiveChunker
# class SlidingWindowChunker

class SimpleChunker:

    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text):

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:

            end = start + self.chunk_size
            chunk = text[start:end]

            chunks.append(chunk)

            start += self.chunk_size - self.overlap

        return chunks