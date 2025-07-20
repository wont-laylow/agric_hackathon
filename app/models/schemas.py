from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    user_id: int
    message: str

class ChatResponse(BaseModel):
    reply: str

class FeedbackCreate(BaseModel):
    rating: bool
    comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    message: str

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    message: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserLoginResponse(BaseModel):
    message: str