from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from app.api.endpoints import router as analyzer_router
from app.api.chat_endpoints import router as chat_router

app = FastAPI(
    title="RepoPilot API",
    description="Repo analyzer backend",
    version="0.1.0"
)

# Enable CORS for frontend client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyzer_router, prefix="/api/analyzer", tags=["Analyzer"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])

@app.get("/")
def read_root():
    return {"message": "Welcome to RepoPilot Backend"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "db_url_set": bool(settings.database_url)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)
