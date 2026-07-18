from fastapi import FastAPI
from .config import settings
from app.api.endpoints import router as analyzer_router

app = FastAPI(
    title="RepoPilot API",
    description="Repo analyzer backend",
    version="0.1.0"
)

app.include_router(analyzer_router, prefix="/api/analyzer", tags=["Analyzer"])

@app.get("/")
def read_root():
    return {"message": "Welcome to RepoPilot Backend"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "db_url_set": bool(settings.database_url)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)
