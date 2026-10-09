from fastapi import FastAPI
from src.service.rag_pipeline import process_input
from pydantic import BaseModel

api = FastAPI()

class ChatRequest(BaseModel):
    message: str

@api.post("/chat")
def chat(message: ChatRequest):
    response = process_input(message.message)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="localhost", port=8000)