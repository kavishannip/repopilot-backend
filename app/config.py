from dotenv import load_dotenv
load_dotenv(override=True)

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────
    database_url: str = ""

    # ── GitHub ───────────────────────────────────────────────
    github_token: str = ""
    github_raw_concurrency: int = 8

    # ── OpenAI Embeddings ────────────────────────────────────
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    # Chunks per embedding request (OpenAI handles up to 2048 inputs per request)
    embed_batch_size: int = 100
    # RPM limit for OpenAI API (Tier 1 is 3,000+ RPM)
    embed_rpm_limit: int = 3000
    # Minimum delay between consecutive embedding requests
    embed_min_delay_seconds: float = 0.05

    # ── NVIDIA NIM (LLM Reasoning) ───────────────────────────
    nvidia_api_key: str = ""
    # LLM: nemotron-3-super-120b-a12b — 1M ctx, LatentMoE, reasoning mode, RAG-optimized
    repo_analizer_model: str = "nvidia/nemotron-3-super-120b-a12b"
    # Enable chain-of-thought reasoning on final synthesis pass (slower but more accurate)
    llm_enable_thinking: bool = True
    # Context budget for final synthesis — 1M ctx window means we can be generous
    llm_max_tokens: int = 16384   # output token cap (generous for rich structured JSON)
    llm_context_chars: int = 60_000  # total input context chars to feed final synthesis
    
    # ── Smart File Selection ─────────────────────────────────
    smart_file_select_enabled: bool = True
    smart_file_select_max_files: int = 100

    # ── Groq (Phase 3 group summaries) ──────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"    # swap via GROQ_MODEL env var to experiment
    # Semaphore limit for Phase 3 parallel calls — tuned for Groq free-tier ~6K TPM
    phase3_concurrency: int = 2

    # ── Unused (Groq replaced by NVIDIA NIM) ─────────────────
    supabase_url: str = ""
    supabase_key: str = ""

    # ── Analysis tuning ──────────────────────────────────────
    enable_graph_analysis: bool = True   # Set to False via env ENABLE_GRAPH_ANALYSIS=false to disable graphs
    max_file_content_chars: int = 8000   # per-file read cap for critical files
    max_repo_size_kb: int = 100 * 1024   # 100 MB repo size cap
    analysis_timeout_seconds: int = 600  # 10 min total timeout
    max_source_file_chars: int = 200_000  # per-file cap when chunking selected source files
    repo_cache_expire_days: int = 30     # How many days before repo data cache expires and is deleted from DB

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
