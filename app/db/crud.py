from sqlalchemy.ext.asyncio import AsyncSession
from .models import PredictionLog

async def create_prediction_log(
    db: AsyncSession,
    *,
    image_filename: str,
    predicted_class: str,
    predicted_crop: str,
    predicted_disease: str,
    confidence: float,
    is_healthy: bool
) -> PredictionLog:
    """
    Creates and saves a new prediction log entry in the database.

    Args:
        db: The database session.
        image_filename: The name of the uploaded file.
        predicted_class: The raw class name from the model.
        predicted_crop: The parsed crop name.
        predicted_disease: The parsed disease name.
        confidence: The model's confidence score.
        is_healthy: A boolean indicating if the prediction is "healthy".

    Returns:
        The newly created PredictionLog object.
    """
    db_log = PredictionLog(
        image_filename=image_filename,
        predicted_class=predicted_class,
        predicted_crop=predicted_crop,
        predicted_disease=predicted_disease,
        confidence=confidence,
        is_healthy=is_healthy
    )
    
    db.add(db_log)
    await db.commit()
    
    await db.refresh(db_log)
    
    return db_log