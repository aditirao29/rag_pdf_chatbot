from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.chunking import clean_text,chunk_text
from embeddings.embedder import Embedder
from vector_store.faiss_store import FAISSStore
from rag.pipeline import RAGPipeline
from rag.full_pipeline import FullRAGPipeline
from llm.ollama_client import OllamaClient

text = extract_text_from_pdf("pjo1.pdf")
"""print(text[:500])
print("\nLength: ",len(text))"""
cleaned = clean_text(text)
chunks = chunk_text(cleaned)
"""print("Number of chunks:", len(chunks))
print("\nFirst chunk:\n", chunks[0])"""

embedder = Embedder()
embeddings = embedder.embed(chunks)
# print("Embeddings ready")

dimension = embeddings.shape[1]
store = FAISSStore(dimension)
store.add_embeddings(embeddings,chunks)
# print("Stored embeddings")

rag = RAGPipeline(embedder,store)
llm = OllamaClient()
chatbot = FullRAGPipeline(rag,llm)
answer = chatbot.ask("What did Percy do with Medusa's head?")
print("\nAnswer:\n")
print(answer)
"""context = rag.run("Who is Percy Jackson?")
print("\nRetrieved context:\n")
print(context[:500])"""

"""query_embedding = embedder.embed(query)[0]
results = store.search(query_embedding,k=3)
print("\nTop Results:\n")
for i in results:
    print(i[:200])
    print("---------")"""