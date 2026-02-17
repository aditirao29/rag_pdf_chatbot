from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self,model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded")
    
    def embed(self,texts):
        if isinstance(texts,str):
            texts = [texts]
        return self.model.encode(texts,show_progress_bar=False)
