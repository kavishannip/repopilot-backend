from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.api.endpoints import router as analyzer_router

app = FastAPI(
    title="RepoPilot API",
    description="Repo analyzer backend (LangChain Agent)",
    version="0.2.0"
)

app.include_router(analyzer_router, prefix="/api/analyzer", tags=["Analyzer"])

@app.get("/")
def read_root():
    return {"message": "Welcome to RepoPilot Backend"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)
