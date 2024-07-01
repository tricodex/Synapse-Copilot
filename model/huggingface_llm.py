import aiohttp
import os
from langchain.llms.base import BaseLLM
from pydantic import Field

HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")
DEFAULT_HUGGINGFACE_MODEL = os.getenv("DEFAULT_HUGGINGFACE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

class HuggingFaceLLM(BaseLLM):
    model_id: str = Field(default=DEFAULT_HUGGINGFACE_MODEL)
    api_url: str = Field(init=False)
    headers: dict = Field(init=False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        self.headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    
    async def _call(self, inputs: str) -> str:
        async with aiohttp.ClientSession() as session:
            payload = {"inputs": inputs}
            async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                result = await response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', '').strip()
        return str(result)

    def _generate(self, inputs: str) -> str:
        import asyncio
        return asyncio.run(self._call(inputs))

    @property
    def _llm_type(self) -> str:
        return "huggingface"