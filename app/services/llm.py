"""
LLM factory for RepoPilot.

Model: nvidia/nemotron-3-super-120b-a12b via NVIDIA NIM API
  - 120B total / 12B active (LatentMoE — Mamba-2 + MoE + Attention hybrid)
  - Up to 1M token context window
  - Multi-Token Prediction (MTP) for faster generation
  - Reasoning mode via .with_thinking_mode(enabled=True/False)
  - Optimized for RAG, agentic workflows, long-context reasoning

Strategy:
  - Fast pass (group summaries): thinking OFF — speed matters, simple summarization task
  - Final synthesis pass: thinking ON — complex architecture reasoning benefits from CoT
"""

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from app.config import settings


def get_llm(
    temperature: float = 0.0,
    max_tokens: int = None,
) -> ChatNVIDIA:
    """
    Base factory — returns ChatNVIDIA without thinking mode set.
    Use get_fast_llm() or get_reasoning_llm() for pipeline passes.
    """
    if not settings.nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY is not set in environment or config.")

    return ChatNVIDIA(
        api_key=settings.nvidia_api_key,
        model=settings.repo_analizer_model,
        temperature=temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
    )


def get_fast_llm() -> ChatNVIDIA:
    """
    Group summaries pass — thinking OFF, lower token cap.
    Fast inference: LatentMoE only activates 12B/120B params.
    """
    return ChatNVIDIA(
        api_key=settings.nvidia_api_key,
        model=settings.repo_analizer_model,
        temperature=0.0,
        max_tokens=2048,
        timeout=60, # 1 minute timeout for fast pass
    ).with_thinking_mode(enabled=False)


def get_reasoning_llm() -> ChatNVIDIA:
    """
    Final synthesis pass — thinking ON (if enabled in settings), generous token cap.
    Chain-of-thought reasoning for architecture analysis with 1M context window.
    """
    base = ChatNVIDIA(
        api_key=settings.nvidia_api_key,
        model=settings.repo_analizer_model,
        temperature=0.0,
        max_tokens=settings.llm_max_tokens,
        timeout=300, # 5 minute timeout since thinking mode takes longer to generate
    )
    return base.with_thinking_mode(enabled=settings.llm_enable_thinking)
