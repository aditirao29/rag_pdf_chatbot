import os

from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.chunking import chunk_pages
from embeddings.embedder import Embedder
from vector_store.faiss_store import FAISSStore
from rag.pipeline import RAGPipeline
from rag.full_pipeline import FullRAGPipeline
from llm.ollama_client import OllamaClient

print("Loading all PDFs...")

pdf_folder = "data"
all_chunks = []

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        path = os.path.join(pdf_folder, file)
        print("Processing:", path)
        pages = extract_text_from_pdf(path)
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)
print("Total chunks:", len(all_chunks))

embedder = Embedder()
chunk_texts = [c["text"] for c in all_chunks]
embeddings = embedder.embed(chunk_texts)

dimension = embeddings.shape[1]
store = FAISSStore(dimension)
store.add_embeddings(embeddings, all_chunks)
print("Vector store ready")

rag = RAGPipeline(embedder, store)
llm = OllamaClient()
chatbot = FullRAGPipeline(rag, llm)

result = chatbot.ask("What did Percy do with Medusa's head?")
print("\nAnswer:\n", result["answer"])
print("\nSources:")
for s in result["sources"]:
    print("-", s)