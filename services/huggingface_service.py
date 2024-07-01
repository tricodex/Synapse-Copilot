import aiohttp
import os

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
DEFAULT_HUGGINGFACE_MODEL = os.getenv("DEFAULT_HUGGINGFACE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

class HuggingFaceService:
    async def summarize_text(self, text: str, model_id: str = None):
        model_id = model_id or DEFAULT_HUGGINGFACE_MODEL
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        
        payload = {
            "inputs": text,
            "parameters": {"max_length": 100, "min_length": 30, "do_sample": False}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                result = await response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('summary_text', '').strip()
        return str(result)

    async def analyze_sentiment(self, text: str, model_id: str = None):
        model_id = model_id or "distilbert-base-uncased-finetuned-sst-2-english"
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        
        payload = {"inputs": text}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                result = await response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('label', '').strip()
        return str(result)

    async def generate_text(self, prompt: str, model_id: str = None):
        model_id = model_id or DEFAULT_HUGGINGFACE_MODEL
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {"max_length": 150, "do_sample": True}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                result = await response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', '').strip()
        return str(result)

    async def answer_question(self, question: str, context: str, model_id: str = None):
        model_id = model_id or "deepset/roberta-base-squad2"
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        
        payload = {
            "inputs": {
                "question": question,
                "context": context
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                result = await response.json()
        
        if isinstance(result, dict):
            return result.get('answer', '').strip()
        return str(result)
