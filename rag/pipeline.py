class RAGPipeline:
    def __init__(self,embedder,vector_store):
        self.embedder = embedder
        self.vector_store = vector_store
    
    def retrieve_context(self,query,k=3):
        """query = retrieve top k chunks"""
        query_embeddings = self.embedder.embed(query)[0]
        results = self.vector_store.search(query_embeddings,k=k)
        return results
    
    def build_context(self,chunks):
        """combine retrieved chunks into single context string"""
        context_parts = []
        for i in chunks:
            context_parts.append(f"[Source: {i['source']} Page:{i['page']}]\n{i['text']}")
        return "\n\n".join(context_parts)
        
    def run(self,query,k=3):
        """
        full pipeline:
        query -> retrieve -> build context
        """
        chunks = self.retrieve_context(query,k=k)
        context = self.build_context(chunks)
        return context