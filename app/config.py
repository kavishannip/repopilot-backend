from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    groq_api_key: str = ""
    github_token: str = ""
    database_url: str = ""
    nvidia_api_key: str = ""
    repo_analizer_model: str = "llama-3.3-70b-versatile"
    # Optional ones
    supabase_url: str = ""
    supabase_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
