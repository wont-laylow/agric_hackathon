from fastapi import APIRouter, Request, UploadFile, File, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
import base64
import traceback

from ..db.database import get_db_session
from ..ml.inference import run_prediction, is_image_a_leaf_permissive
from ..schemas import PredictionResponse, ChatRequest, ChatResponse
from ..chatbot.local_service import get_bot_service, LocalKnowledgeBot

router = APIRouter()
api_router = APIRouter(prefix="/api")
templates = Jinja2Templates(directory="app/templates")

@router.get("/", include_in_schema=False)
async def get_upload_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/predict", include_in_schema=False)
async def create_prediction(request: Request, file: UploadFile = File(...), db: AsyncSession = Depends(get_db_session)):
    try:
        image_bytes = await file.read()

        if not is_image_a_leaf_permissive(image_bytes):
            error_message = (
                "Validation Failed: The uploaded image does not appear to be a plant leaf. "
                "Please upload a clear, close-up photo of a crop leaf."
            )
            return templates.TemplateResponse("index.html", {"request": request, "error": error_message})

        top_results = run_prediction(image_bytes)
        top_prediction_data = top_results[0]
        confidence = top_prediction_data['confidence']
        predicted_class = top_prediction_data['predicted_class'].replace(' ', '_')

        parts = predicted_class.split('_')
        crop, disease = parts[0], " ".join(parts[1:]).title()
        is_healthy = "healthy" in disease.lower()
        if is_healthy: 
            disease = "Healthy"

        prediction_data = PredictionResponse(
            filename=file.filename, 
            content_type=file.content_type, 
            predicted_class=predicted_class, 
            crop=crop, 
            disease=disease, 
            confidence=confidence, 
            is_healthy=is_healthy, 
            top_predictions=top_results,
        )
        
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        response_data = {
            "request": request, 
            "prediction": prediction_data.dict(), 
            "image_b64": image_b64,
        }
        
        return templates.TemplateResponse("index.html", response_data)
            
    except Exception as e:
        print(f"ERROR: Unexpected error in prediction endpoint: {e}")
        traceback.print_exc()
        return templates.TemplateResponse("index.html", {"request": request, "error": f"A critical server error occurred: {str(e)}"})

@api_router.post("/chat", response_model=ChatResponse)
async def handle_chat(
    chat_request: ChatRequest,
    chatbot: LocalKnowledgeBot = Depends(get_bot_service)
):
    disease_context = chatbot.find_best_match(chat_request.predicted_class)
    
    if not disease_context:
        return ChatResponse(response="I'm sorry, I could not find detailed information for that disease in my knowledge base.")

    answer = chatbot.format_answer(chat_request.query, disease_context)
    
    return ChatResponse(response=answer)