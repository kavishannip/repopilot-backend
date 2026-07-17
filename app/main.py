from fastapi import FastAPI
from .config import settings

app = FastAPI(
    title="RepoPilot API",
    description="Repo analyzer backend",
    version="0.1.0"
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "db_url_set": bool(settings.database_url)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)
