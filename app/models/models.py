# models.py
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from .db import Base
from datetime import datetime
from sqlalchemy import DateTime

# Users
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)

    chat_history = relationship("ChatHistory", back_populates="user")
    login = relationship("UserLogin", back_populates="user")

class UserLogin(Base):
    __tablename__ = "user_login"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    username = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="login")
    

# Chat
class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    question = Column(String)
    response = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="chat_history")
    feedback = relationship("Feedback", back_populates="chat")

class Feedback(Base):
    __tablename__ = "chat_feedback"  

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chat_history.id"))
    rating = Column(Boolean)
    comment = Column(String)

    chat = relationship("ChatHistory", back_populates="feedback")

# Finetune requests

# Image upload