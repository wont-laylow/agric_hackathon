from fastapi import FastAPI
from app.routes import chat, users

app = FastAPI(title="ToutBot Backend ")

app.include_router(chat.router)
app.include_router(users.router)

@app.get("/health")
async def health():
    return {"message": "ToutBot API is running"}
