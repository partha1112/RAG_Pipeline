from fastapi import FastAPI
from src.utils.exceptions import PoorRetrievalQualityException
from src.service.rag_pipeline import process_input
from pydantic import BaseModel

from src.utils.schema import UserInput

api = FastAPI()

@api.post("/chat")
def chat(userInput: UserInput):
    try: 
        response = process_input(userInput)
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