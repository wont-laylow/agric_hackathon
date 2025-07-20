from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import UserCreate, UserResponse, UserLogin, UserLoginResponse
from app.models.db import get_db
from sqlalchemy.orm import Session
from app.models.model import User
from utils.logger import logger
import uuid

router = APIRouter(prefix="/api/v1/users", tags=["Auth"])

def generate_token():
    return str(uuid.uuid4())

@router.post("/register", response_model=UserResponse)
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


@router.post("/login", response_model=UserLoginResponse)
async def login_user(user: UserLogin, db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(
            (User.username == user.username) & (User.password == user.password)).first()
        if not db_user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = generate_token()
        db_user.token = token
        db.commit()
        db.refresh(db_user)

        return {"message": "User logged in successfully", "token": token}
    except Exception as e:
        logger.error(f"Error in login_user: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

