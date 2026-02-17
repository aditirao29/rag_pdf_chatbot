from llm.ollama_client import OllamaClient

llm = OllamaClient()
response = llm.generate("Who is Percy Jackson's godly parent?")
print(response)