from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class ChatRequest(BaseModel):
    """Defines the structure for a request to the chatbot API."""
    query: str = Field(..., example="What are the symptoms?")
    predicted_class: str = Field(..., example="Maize_leaf_blight")

class ChatResponse(BaseModel):
    """Defines the structure for a response from the chatbot API."""
    response: str = Field(..., example="The main effects and symptoms are...")

class SinglePrediction(BaseModel):
    """Defines the data for a single item in the prediction list."""
    predicted_class: str
    confidence: float

class PredictionResponse(BaseModel):
    """
    Defines the data structure for the prediction response sent to the frontend.
    """
    filename: str
    content_type: str
    
    predicted_class: str = Field(..., example="Tomato_leaf_blight")
    crop: str = Field(..., example="Tomato")
    disease: str = Field(..., example="Leaf Blight")
    confidence: float = Field(..., gt=0, le=100, example=95.43)
    is_healthy: bool
    
    top_predictions: List[SinglePrediction]
    confidence_warning: Optional[str] = Field(None, example="Low Confidence (22.74%): The model is not confident...")

    class Config:
        from_attributes = True