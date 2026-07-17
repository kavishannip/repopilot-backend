# RepoPilot Backend

FastAPI backend for analyzing public GitHub repositories.

## Setup

1. Create a virtual environment and activate it:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Configure environment variables in `.env`:
   - `DATABASE_URL`
   - `GROQ_API_KEY`
   - `GITHUB_TOKEN`

## Run

Run the backend on port 5000:
```powershell
python -m app.main
```
Or directly with uvicorn:
```powershell
uvicorn app.main:app --port 5000 --reload
```
Check health endpoint at: http://localhost:5000/health
