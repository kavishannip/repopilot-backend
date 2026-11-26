# RepoPilot Backend

FastAPI service for [RepoPilot](https://github.com/kavishannip/repopilot-frontend), an AI-powered tool for analyzing public GitHub repositories and answering questions about their code. The backend exposes analyzer and chat APIs to the Next.js frontend, fetches repository data from GitHub, indexes source code for semantic retrieval, and stores analysis results and chat history.

## End-to-end analysis pipeline

1. **Quick summary:** The frontend submits a public GitHub URL to `POST /api/analyzer/quicksummary`. The backend validates the URL, fetches repository metadata and key files such as the README and manifests, generates a concise summary, and caches the result in PostgreSQL.
2. **Submit deep analysis:** The frontend requests `POST /api/analyzer/analyze` with the repository ID. The backend returns a job ID immediately and runs the analysis in a background task; the frontend polls `GET /api/analyzer/status/{job_id}` for progress and the result.
3. **Fetch, select, and map files:** The pipeline reads the repository tree and commit information from GitHub, chooses relevant files for in-depth analysis, and optionally scans source code to build file and import/dependency relationships.
4. **Chunk and index code:** Source files are split into code-aware chunks (Tree-sitter for supported Python/JavaScript/TypeScript syntax, with language-aware and plain-text fallbacks). OpenAI embeddings are generated in batches, then chunks, metadata, and vectors are persisted in PostgreSQL with pgvector.
5. **Summarize and synthesize:** Static repository structure and key files are combined with targeted vector-search results. The service generates summaries for architectural areas and synthesizes them into a structured repository brief. Results and intermediate summaries are cached by commit so unchanged repositories can reuse work.
6. **Explore and chat:** The API can generate a Mermaid architecture diagram from the brief. For chat, it retrieves relevant code chunks, uses an LLM to answer from that context, and can select and index additional candidate files when the initial context is insufficient. Chat sessions and messages are persisted.
7. **Keep the index current:** Staleness checks compare the cached commit with GitHub. Sync updates changed or added files and removes stale file chunks, avoiding a full re-index when incremental processing is possible.

## Technology

- Python, FastAPI, and Uvicorn
- PostgreSQL, SQLAlchemy async, and pgvector
- GitHub REST API for repository metadata, trees, commits, and file contents
- Tree-sitter and LangChain text splitters for code chunking
- OpenAI `text-embedding-3-small` embeddings
- Groq for fast/group-summary tasks and NVIDIA NIM for reasoning/synthesis (model configuration is in `app/config.py`)
- Asgardeo JWT verification for authenticated API requests

## Requirements

- Python 3.10+ (use a Python version compatible with the pinned Tree-sitter dependencies)
- PostgreSQL with the pgvector extension available
- API credentials for GitHub, OpenAI embeddings, Groq, NVIDIA NIM, and Asgardeo authentication

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configure a `.env` file in the project root. Settings are loaded from environment variables; the following names correspond to the fields in `app/config.py`:

```dotenv
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/repopilot
GITHUB_TOKEN=your_github_token
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
NVIDIA_API_KEY=your_nvidia_api_key
ASGARDEO_ORG_NAME=your_asgardeo_organization
ASGARDEO_CLIENT_ID=your_asgardeo_client_id
```

Use a database URL compatible with SQLAlchemy's async PostgreSQL driver. Ensure the database has pgvector installed/enabled; the scripts under `scripts/db/` include database initialization and schema utilities. Keep real credentials out of source control.

## Run locally

Start the API on port 5000:

```bash
python -m app.main
```

Or use Uvicorn with auto-reload:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

Check `http://localhost:5000/health`. Interactive API documentation is available at `http://localhost:5000/docs`.

## Main API routes

- `/api/analyzer` — quick summary, deep-analysis jobs and status, staleness checks, sync, repository details, and architecture diagrams
- `/api/chat` — create chat sessions, send questions, retrieve history, list sessions, and delete sessions
- `/health` — service health check

The frontend defaults to `http://localhost:5000`; configure `NEXT_PUBLIC_API_URL` in the frontend when the backend runs at a different address. The backend CORS configuration must also allow the frontend's origin in deployed environments.

## Limits and operational notes

Analysis has repository size/file-count limits and may be constrained by GitHub API rate limits, model context limits, and language-specific parsing coverage. See [LIMITATIONS.md](LIMITATIONS.md) for known limitations. Analysis jobs are recorded in the database, while execution is initiated as an in-process background task; run and scale the service with that deployment model in mind.

## Project scripts

- `python scripts/db/init_db.py` — initialize database tables, pgvector extensions, and indexes
- `python scripts/db/migrate_schema.py` — run schema migrations
- `python scripts/db/delete_repo.py` — delete repository data
- `python scripts/db/add_col.py` — add a column to the repositories table
- `scripts/test_scripts/` — ad-hoc integration and LLM verification scripts

## Related repository

- [RepoPilot Frontend](https://github.com/kavishannip/repopilot-frontend) — Next.js user interface
