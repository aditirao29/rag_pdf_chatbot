class FullRAGPipeline:
    def __init__(self, rag_pipeline, llm_client):
        self.rag = rag_pipeline
        self.llm = llm_client

    def build_prompt(self, context, question):
        return f"""
    You are a helpful and friendly assistant.

    Use the information below to answer the question clearly and naturally.

    Do NOT mention "context", "provided text", or "information above".
    Just answer like a normal chatbot.

    If the answer cannot be found, say:
    "I’m not sure about that."

    Information:
    {context}

    Question:
    {question}

    Answer:
    """

    def ask(self, question):
        chunks = self.rag.retrieve_context(question, k=10)
        context = self.rag.build_context(chunks)

        prompt = self.build_prompt(context, question)
        answer = self.llm.generate(prompt)

        sources = [
            f"{c['source']} page {c['page']}"
            for c in chunks
        ]
        print("\n--- RETRIEVED CHUNKS ---")
        for c in chunks:
            print(c["text"][:200])

        return {
            "answer": answer,
            "sources": list(set(sources))
        }
