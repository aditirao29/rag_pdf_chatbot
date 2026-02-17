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
        context = self.rag.run(question, k=3)
        prompt = self.build_prompt(context, question)

        response = self.llm.generate(prompt)
        return response