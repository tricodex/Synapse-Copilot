from fastapi import FastAPI
from model.api_llm import ApiLLM
from services.huggingface_service import HuggingFaceService
from model.huggingface_llm import HuggingFaceLLM
from utils.utils import ReducedOpenAPISpec

app = FastAPI()

huggingface_service = HuggingFaceService()
llm = HuggingFaceLLM()  

# Mock OpenAPI Spec for testing
class MockOpenAPISpec:
    def __init__(self):
        self.endpoints = [("GET", "/mock_endpoint")]

api_spec = MockOpenAPISpec()  # Replace with actual ReducedOpenAPISpec instance

api_llm = ApiLLM(llm=llm, api_spec=api_spec, scenario="tmdb", requests_wrapper=None)

@app.post("/query")
async def query_api(query: str):
    result = await api_llm._call({"query": query})
    return result


# Use the Hugging Face LLM

# api_llm = ApiLLM(llm=llm, api_spec=None, scenario="tmdb", requests_wrapper=None)

# @app.post("/query")
# async def query_api(query: str):
#     result = await api_llm._call({"query": query})
#     return result
