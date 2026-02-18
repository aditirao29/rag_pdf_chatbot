from rank_bm25 import BM25Okapi

class BM25Store:
    def __init__(self):
        self.corpus = []
        self.metadata = []
        self.bm25 = None

    def add_documents(self, chunks):
        if not chunks:
            self.corpus = []
            self.metadata = []
            self.bm25 = None
            return
        
        texts = [c["text"] for c in chunks]
        tokenized = [t.lower().split() for t in texts]

        self.corpus = tokenized
        self.metadata = chunks
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, k=10):
        if not self.bm25:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [self.metadata[i] for i, _ in ranked[:k]]