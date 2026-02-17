from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.chunking import chunk_pages
from embeddings.embedder import Embedder
from vector_store.faiss_store import FAISSStore
from rag.pipeline import RAGPipeline
from rag.full_pipeline import FullRAGPipeline
from llm.ollama_client import OllamaClient
import os

print("Setting up chatbot...")
all_chunks = []
pdf_folder = "data"

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        path = os.path.join(pdf_folder, file)
        pages = extract_text_from_pdf(path)
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)

print("Total chunks:", len(all_chunks))

chunk_texts = [c["text"] for c in all_chunks]
embedder = Embedder()
embeddings = embedder.embed(chunk_texts)

store = FAISSStore(embeddings.shape[1])
store.add_embeddings(embeddings, all_chunks)

rag = RAGPipeline(embedder, store)
llm = OllamaClient()
chatbot = FullRAGPipeline(rag, llm)

print("Chatbot ready.\n")

tests = [
    {"question": "Who is Percy Jackson?", "expected_keywords": ["Poseidon", "demigod"]},
    {"question": "What is Camp Half-Blood?", "expected_keywords": ["camp", "demigod"]},
    {"question": "Who is Grover?", "expected_keywords": ["satyr", "friend"]},
    {"question": "Where did Percy study?", "expected_keywords": ["Yancy", "academy"]},
    {"question": "Who is Percy's mother?", "expected_keywords": ["mother"]},
    {"question": "What happens to Percy's teacher?", "expected_keywords": ["teacher"]},
    {"question": "Who is Annabeth?", "expected_keywords": ["Athena"]},
    {"question": "What kind of being is Percy?", "expected_keywords": ["demigod"]},
    {"question": "Where does Percy go for training?", "expected_keywords": ["Camp"]},
    {"question": "Who is Percy's godly parent?", "expected_keywords": ["Poseidon"]}
]

print("Running evaluation...\n")
score = 0
for i, test in enumerate(tests, 1):
    print(f"Test {i}: {test['question']}")
    result = chatbot.ask(test["question"])
    answer = result["answer"]
    print("Answer:", answer)
    print("Sources:", result["sources"], "\n")
    matches = sum(
        keyword.lower() in answer.lower()
        for keyword in test["expected_keywords"]
    )
    required = len(test["expected_keywords"]) // 2 + 1
    if matches >= required:
        print("✅ PASS\n")
        score += 1
    else:
        print("❌ FAIL\n")

print("=" * 40)
print("Final Accuracy:", score / len(tests))
print("=" * 40)