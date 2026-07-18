from langchain_groq import ChatGroq
from app.config import settings

def get_llm(model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.0) -> ChatGroq:
    """
    Factory function to get a configured ChatGroq instance.
    Defaults to LLaMA3 70B for high quality summarization.
    """
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in environment or config.")
        
    return ChatGroq(
        api_key=settings.groq_api_key,
        model_name=model_name,
        temperature=temperature
    )
