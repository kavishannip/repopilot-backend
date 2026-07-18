from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────
    database_url: str = ""

    # ── GitHub ───────────────────────────────────────────────
    github_token: str = ""

    # ── NVIDIA NIM (used for both embeddings AND LLM) ────────
    nvidia_api_key: str = ""
    # Embedding model (nemotron-3-embed-1b, 2048-dim vectors)
    nvidia_embed_model: str = "nvidia/nemotron-3-embed-1b"
    # LLM: nemotron-3-super-120b-a12b — 1M ctx, LatentMoE, reasoning mode, RAG-optimized
    repo_analizer_model: str = "nvidia/nemotron-3-super-120b-a12b"
    # Enable chain-of-thought reasoning on final synthesis pass (slower but more accurate)
    llm_enable_thinking: bool = True
    # Context budget for final synthesis — 1M ctx window means we can be generous
    llm_max_tokens: int = 16384   # output token cap (generous for rich structured JSON)
    llm_context_chars: int = 60_000  # total input context chars to feed final synthesis

    # ── Unused (Groq replaced by NVIDIA NIM) ─────────────────
    groq_api_key: str = ""         # no longer used for LLM calls
    supabase_url: str = ""
    supabase_key: str = ""

    # ── Analysis tuning ──────────────────────────────────────
    max_file_content_chars: int = 8000   # per-file read cap for critical files
    max_repo_size_kb: int = 100 * 1024   # 100 MB repo size cap
    analysis_timeout_seconds: int = 600  # 10 min total timeout

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
