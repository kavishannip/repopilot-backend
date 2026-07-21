"""
LLM factory for RepoPilot.

Phase 3 (group summaries): Groq — llama-3.1-8b-instant (fast, generous free tier)
Phase 4 (final synthesis): NVIDIA NIM — nemotron-3-super-120b-a12b (reasoning, 1M ctx)

Swap Groq model anytime via GROQ_MODEL env var without code changes.
"""

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from app.config import settings
import logging

logger = logging.getLogger(__name__)

def get_fast_llm() -> ChatNVIDIA:
    """
    Phase 3 (group summaries): NVIDIA NIM (Fast model).
    Replaces Groq. Uses a smaller, faster model (Llama 3.1 8B) hosted on NVIDIA NIM 
    to handle concurrent map-reduce summarization tasks efficiently without hitting 
    the heavy rate limits of the 120B reasoning model.
    """
    if not settings.nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY is not set. Required for Phase 3 group summaries.")

    fast_model = "meta/llama-3.1-8b-instruct"
    logger.debug(f"[LLM] Phase 3 using fast NVIDIA model: {fast_model}")
    return ChatNVIDIA(
        api_key=settings.nvidia_api_key,
        model=fast_model,
        temperature=0.0,
        max_tokens=2048,
    )


def get_reasoning_llm() -> ChatNVIDIA:
    """
    Phase 4 (final synthesis): NVIDIA NIM.
    Reasoning mode (CoT), 1M context window, RAG-optimized.
    Unchanged from original design — this is where output quality comes from.
    """
    if not settings.nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY is not set. Required for Phase 4 synthesis.")

    logger.debug(f"[LLM] Phase 4 using NVIDIA model: {settings.repo_analizer_model}")
    base = ChatNVIDIA(
        api_key=settings.nvidia_api_key,
        model=settings.repo_analizer_model,
        temperature=0.0,
        max_tokens=settings.llm_max_tokens,
        timeout=300,  # thinking mode takes longer
    )
    return base.with_thinking_mode(enabled=settings.llm_enable_thinking)


def get_llm(temperature: float = 0.0, max_tokens: int = None) -> ChatNVIDIA:
    """
    Base NVIDIA factory. Use get_fast_llm() or get_reasoning_llm() for pipeline passes.
    """
    if not settings.nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY is not set.")
    return ChatNVIDIA(
        api_key=settings.nvidia_api_key,
        model=settings.repo_analizer_model,
        temperature=temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
    )
