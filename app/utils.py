from fastapi import Header, HTTPException
from app.models.model import User
from app.models.db import get_db
from sqlalchemy.orm import Session
from fastapi import Depends

def get_current_user(token: str = Header(...), db: Session = Depends(get_db)) -> User:
    user = db.query(User).filter(User.token == token).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return user