import requests

class OllamaClient:
    def __init__(self,model="mistral"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"
    
    def generate(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(self.url, json=payload)
        data = response.json()
        print("LLM response:", data.get("response"))
        return data.get("response", "No response field found")