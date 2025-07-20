from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from .core.config import settings
from .db.database import create_db_and_tables
from .ml.inference import load_models, unload_models 
from .chatbot.local_service import get_bot_service
from .api.endpoints import router as page_router, api_router

# from finetuning.api_endpoints import router as finetuning_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown."""
    print("INFO:     Starting up application...")
    await create_db_and_tables()
    load_models(settings.MODEL_PATH)
    get_bot_service()
    yield
    print("INFO:     Shutting down application...")
    unload_models()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(page_router)
app.include_router(api_router)
# app.include_router(finetuning_router)