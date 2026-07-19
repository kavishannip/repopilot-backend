from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────
    database_url: str = ""

    # ── GitHub ───────────────────────────────────────────────
    github_token: str = ""
    github_raw_concurrency: int = 8

    # ── NVIDIA NIM (used for both embeddings AND LLM) ────────
    nvidia_api_key: str = ""
    # Embedding model (nemotron-3-embed-1b, 2048-dim vectors)
    nvidia_embed_model: str = "nvidia/nemotron-3-embed-1b"
    # Chunks per embedding request — smaller = safer on free tier, but nemotron-3-embed-1b supports 50
    embed_batch_size: int = 10
    # RPM limit for NVIDIA API (free tier is 40)
    embed_rpm_limit: int = 35
    # LLM: nemotron-3-super-120b-a12b — 1M ctx, LatentMoE, reasoning mode, RAG-optimized
    repo_analizer_model: str = "nvidia/nemotron-3-super-120b-a12b"
    # Enable chain-of-thought reasoning on final synthesis pass (slower but more accurate)
    llm_enable_thinking: bool = True
    # Context budget for final synthesis — 1M ctx window means we can be generous
    llm_max_tokens: int = 16384   # output token cap (generous for rich structured JSON)
    llm_context_chars: int = 60_000  # total input context chars to feed final synthesis
    
    # ── Smart File Selection ─────────────────────────────────
    smart_file_select_enabled: bool = True
    smart_file_select_max_files: int = 50

    # ── Groq (Phase 3 group summaries) ──────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"    # swap via GROQ_MODEL env var to experiment
    # Semaphore limit for Phase 3 parallel calls — tuned for Groq free-tier ~6K TPM
    phase3_concurrency: int = 2

    # ── Unused (Groq replaced by NVIDIA NIM) ─────────────────
    supabase_url: str = ""
    supabase_key: str = ""

    # ── Analysis tuning ──────────────────────────────────────
    max_file_content_chars: int = 8000   # per-file read cap for critical files
    max_repo_size_kb: int = 100 * 1024   # 100 MB repo size cap
    analysis_timeout_seconds: int = 600  # 10 min total timeout
    max_source_file_chars: int = 200_000  # per-file cap when chunking selected source files
    repo_cache_expire_days: int = 30     # How many days before repo data cache expires and is deleted from DB

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
