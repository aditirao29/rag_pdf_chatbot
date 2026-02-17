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
        print("Raw response:", response.json())
        return response.json().get("response", "No response field found")