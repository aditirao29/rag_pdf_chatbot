class HybridRAGPipeline:
    def __init__(self, embedder, faiss_store, bm25_store):
        self.embedder = embedder
        self.faiss = faiss_store
        self.bm25 = bm25_store

    def retrieve_context(self, query, k=8):
        query_embedding = self.embedder.embed(query)[0]

        dense = self.faiss.search(query_embedding, k=10)
        sparse = self.bm25.search(query, k=10)

        scores = {}

        for i, c in enumerate(dense):
            key = (c["source"], c["page"], c["text"])
            scores[key] = scores.get(key, 0) + (10 - i)

        for i, c in enumerate(sparse):
            key = (c["source"], c["page"], c["text"])
            scores[key] = scores.get(key, 0) + (10 - i)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for key,_ in ranked[:k]:
            for c in dense + sparse:
                if (c["source"], c["page"], c["text"]) == key:
                    results.append(c)
                    break

        return results


    def build_context(self, chunks):
        context_parts = []
        for i in chunks:
            context_parts.append(
                f"[Source: {i['source']} Page:{i['page']}]\n{i['text']}"
            )
        return "\n\n".join(context_parts)