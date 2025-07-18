from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import UserCreate, UserResponse, UserLogin, UserLoginResponse
from app.models.db import get_db
from sqlalchemy.orm import Session
from app.models.models import User
from utils.logger import logger

router = APIRouter(prefix="/api/v1", tags=["Users"])

@router.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    try:
        new_user = User(
            username=user.username,
            email=user.email, 
            password=user.password
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"message": "User created successfully"}
    except Exception as e:
        logger.error(f"Error in create_user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/login", response_model=UserLoginResponse)
async def login_user(user: UserLogin, db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(User.username == user.username).first()
   
    return {"message": "User logged in successfully"}

