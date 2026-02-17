import faiss
import numpy as np
import pickle

class FAISSStore:
    def __init__(self,dimension):
        """dimension = embedding vector size (384 for MiniLM)"""
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.text_chunks = []
    
    def add_embeddings(self,embeddings,chunks):
        """
        Docstring for add_embeddings
        
        :param self: Description
        :param embeddings: numpy array
        :param chunks: list of text chunks
        """
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.text_chunks.extend(chunks)
    
    def search(self,query_embedding,k=3):
        """
        Docstring for search
        
        :param self: Description
        :param query_embedding: single vector
        :param k: returns top k matching chunks
        """
        query_embedding = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_embedding)
        distances,indices = self.index.search(query_embedding,k)
        results = [self.text_chunks[i] for i in indices[0]]
        return results
    
    def save(self,index_path="vector_store/index.faiss",chunks_path="vector_store/chunks.pkl"):
        """save index + chunks"""
        faiss.write_index(self.index,index_path)
        with open(chunks_path,"wb") as f:
            pickle.dump(self.text_chunks,f)
        
    def load(self,index_path="vector_store/index.faiss",chunks_path="vector_store/chunks.pkl"):
        """load index + chunks"""
        self.index = faiss.read_index(index_path)
        with open(chunks_path,"rb") as f:
            self.text_chunks = pickle.load(f)