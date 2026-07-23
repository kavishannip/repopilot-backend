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
33: Check health endpoint at: http://localhost:5000/health
34: 
35: ## Project Scripts
36: 
37: Database management and utility scripts are categorized under `scripts/`:
38: - **Database & Schema Scripts** (`scripts/db/`):
39:   - `python scripts/db/init_db.py` - Initialize database tables, pgvector extensions, and indexes.
40:   - `python scripts/db/migrate_schema.py` - Run database schema migrations.
41:   - `python scripts/db/delete_repo.py` - Delete repository data from database.
42:   - `python scripts/db/add_col.py` - Add columns to repositories table.
43: - **Experimental Test Scripts** (`scripts/test_scripts/`):
44:   - Contains ad-hoc integration and LLM verification scripts (e.g. `test_nvidia.py`, `test_groq_embed.py`, `scratch_test.py`).

