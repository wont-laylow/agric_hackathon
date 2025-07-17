from fastapi import FastAPI
from app.routes import chat

app = FastAPI(title="ToutBot Backend ")

app.include_router(chat.router)

@app.get("/health")
async def health():
    return {"message": "ToutBot API is running"}
