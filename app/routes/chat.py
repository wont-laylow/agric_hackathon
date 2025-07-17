from fastapi import APIRouter
from fastapi import HTTPException
from rag.app.toutbot import chat_Toutbot
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter(
    prefix="/api/v1",
    tags=["Chat"]
)

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        user_message = request.message
        response_text = chat_Toutbot(user_message)  
        return ChatResponse(reply=response_text)
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

