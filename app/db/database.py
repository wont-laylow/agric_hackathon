from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from typing import AsyncGenerator

from ..core.config import settings
from .models import Base

engine = create_async_engine(settings.DATABASE_URL, echo=False)

async_session_maker = async_sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

async def create_db_and_tables():
    """
    Initializes the database by creating all tables defined in the Base metadata.
    This is typically run once at application startup.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a new database session per request.
    Ensures the session is always closed after the request is finished.
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()