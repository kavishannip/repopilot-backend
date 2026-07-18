from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str = ""
    github_token: str = ""
    database_url: str = ""
    nvidia_api_key: str = ""
    repo_analizer_model: str = "llama-3.3-70b-versatile"
    # Optional
    supabase_url: str = ""
    supabase_key: str = ""
    # Quality tuning
    max_file_content_chars: int = 8000  # Per-file read cap for critical files
    max_repo_size_kb: int = 100 * 1024  # 100 MB repo size cap
    analysis_timeout_seconds: int = 600  # 10 min total timeout for analysis

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
