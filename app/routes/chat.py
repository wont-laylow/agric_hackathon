from fastapi import APIRouter
from fastapi import HTTPException
from rag.app.toutbot import chat_Toutbot
from app.models.schemas import ChatRequest, ChatResponse
from app.models.schemas import FeedbackCreate, FeedbackResponse
from app.models.db import get_db
from sqlalchemy.orm import Session
from fastapi import Depends
from utils.logger import logger
from app.models.model import ChatHistory, Feedback, User
from datetime import datetime
from app.utils import get_current_user

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])

@router.post("/infer", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        user_message = request.message
        response_text = chat_Toutbot(user_message)  
        
        chat_entry = ChatHistory(
            user_id=request.user_id,
            question=user_message,
            response=response_text,
            timestamp=datetime.utcnow()
        )
        db.add(chat_entry)
        db.commit()

        return ChatResponse(reply=response_text or "No response from the bot")

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackCreate, db: Session = Depends(get_db)):
    try:
        chat_entry = db.query(ChatHistory).filter(ChatHistory.id == feedback.chat_id).first()
        if not chat_entry:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        new_feedback = Feedback(
            rating=feedback.rating,
            comment=feedback.comment
        )

        db.add(new_feedback)
        db.commit()
        db.refresh(new_feedback)
        return {"message": "Feedback submitted successfully"}

    except Exception as e:
        logger.error(f"Error in feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

