from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import shutil

from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.chunking import chunk_pages
from embeddings.embedder import Embedder
from vector_store.faiss_store import FAISSStore
from vector_store.bm25_store import BM25Store
from rag.hybrid_pipeline import HybridRAGPipeline
from rag.full_pipeline import FullRAGPipeline
from llm.ollama_client import OllamaClient

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

if os.path.exists(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def initialize_system():
    """
    Create empty RAG system.
    Nothing is stored on disk.
    Everything resets when app stops.
    """
    print("Setting up chatbot (memory-only mode)...")

    embedder = Embedder()
    faiss_store = FAISSStore(384)
    bm25_store = BM25Store()
    rag = HybridRAGPipeline(embedder, faiss_store, bm25_store)
    llm = OllamaClient()
    chatbot = FullRAGPipeline(rag, llm)

    print("Chatbot ready (no persistence).")

    return embedder, faiss_store, bm25_store, chatbot

embedder, faiss_store, bm25_store, chatbot = initialize_system()

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """Ask question to chatbot."""
    data = request.json
    question = data.get("question") if data else None

    if not question:
        return jsonify({"error": "No question provided"}), 400

    result = chatbot.ask(question)
    return jsonify(result)


@app.route("/upload", methods=["POST"])
def upload_pdf():
    """
    Upload PDF → process → embed → add to memory.
    Data stays only while app runs.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    print("Processing uploaded file:", filepath)

    try:
        pages = extract_text_from_pdf(filepath)
        new_chunks = chunk_pages(pages)

        if not new_chunks:
            return jsonify({"error": "No text found in PDF"}), 400

        texts = [c["text"] for c in new_chunks]
        embeddings = embedder.embed(texts)

        faiss_store.add_embeddings(embeddings, new_chunks)
        bm25_store.add_documents(faiss_store.metadata)

        return jsonify({
            "message": "PDF uploaded and indexed successfully (session only)",
            "chunks_added": len(new_chunks)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)