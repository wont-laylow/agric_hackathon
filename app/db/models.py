from sqlalchemy import (
    Column, 
    Integer, 
    String, 
    Float, 
    Boolean, 
    DateTime,
    func
)
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass

class PredictionLog(Base):
    """
    Represents a single prediction event logged in the database.
    """
    __tablename__ = "prediction_logs"

    id: int = Column(Integer, primary_key=True, index=True)
    image_filename: str = Column(String, index=True)
    
    predicted_class: str = Column(String, index=True)
    predicted_crop: str = Column(String, index=True)
    predicted_disease: str = Column(String, index=True)
    
    confidence: float = Column(Float)
    is_healthy: bool = Column(Boolean)
    
    prediction_timestamp: DateTime = Column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return (f"<PredictionLog(id={self.id}, "
                f"crop='{self.predicted_crop}', "
                f"disease='{self.predicted_disease}')>")