class FullRAGPipeline:
    def __init__(self, rag_pipeline, llm_client):
        self.rag = rag_pipeline
        self.llm = llm_client

    def build_prompt(self, context, question):
        return f"""
You are a helpful assistant.

Answer the question ONLY using the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    def ask(self, question):
        chunks = self.rag.retrieve_context(question, k=5)
        context = self.rag.build_context(chunks)

        prompt = self.build_prompt(context, question)
        answer = self.llm.generate(prompt)

        sources = [
            f"{c['source']} page {c['page']}"
            for c in chunks
        ]

        return {
            "answer": answer,
            "sources": list(set(sources))
        }