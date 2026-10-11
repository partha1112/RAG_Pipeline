from fastapi import FastAPI
from src.utils.exceptions import PoorRetrievalQualityException
from src.service.rag_pipeline import process_input
from pydantic import BaseModel

api = FastAPI()

class ChatRequest(BaseModel):
    message: str

@api.post("/chat")
def chat(message: ChatRequest):
    try: 
        response = process_input(message.message)
        return {"response": response}
    except PoorRetrievalQualityException as e :
        return{
            "message": e.message,
            "error_code": e.error_code,
            "details": e.details,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="localhost", port=8000)