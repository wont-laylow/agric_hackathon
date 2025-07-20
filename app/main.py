# =============================================================================
# FILE: app/main.py (FINAL, WITH LOCAL BOT)
# =============================================================================
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from .core.config import settings
from .db.database import create_db_and_tables

# --- UPDATED IMPORTS FOR LOCAL BOT ---
from .ml.inference import load_models, unload_models 
from .chatbot.local_service import get_bot_service # Import the local service
from .api.endpoints import router as page_router, api_router
# ------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown."""
    print("INFO:     Starting up application...")
    await create_db_and_tables()
    # Pre-load ML models and initialize chatbot service on startup
    load_models(settings.MODEL_PATH)
    get_bot_service() # This will initialize the local bot and build the index
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

# Include both routers
app.include_router(page_router)
app.include_router(api_router)