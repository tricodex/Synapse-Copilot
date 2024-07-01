from pydantic import BaseModel

class HuggingFaceRequest(BaseModel):
    prompt: str
    model_id: str = None

class HuggingFaceResponse(BaseModel):
    result: str
    
    