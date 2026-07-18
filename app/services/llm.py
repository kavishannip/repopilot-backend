from langchain_groq import ChatGroq
from app.config import settings


def get_llm(temperature: float = 0.0, max_tokens: int = 8192) -> ChatGroq:
    """
    Factory function to get a configured ChatGroq instance.
    Uses settings.repo_analizer_model for the model name.
    max_tokens defaults to 8192 — important for large structured output responses.
    """
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in environment or config.")

    return ChatGroq(
        api_key=settings.groq_api_key,
        model_name=settings.repo_analizer_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
