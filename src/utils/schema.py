from pydantic import BaseModel

class ScopeModel(BaseModel):
    label: str

class UserInput(BaseModel) :
    message : str
    user_id : int