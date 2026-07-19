import asyncio
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from app.config import settings
from langchain_core.messages import SystemMessage, HumanMessage

async def test_llm():
    system_prompt = """You are an expert software architect producing a comprehensive repository brief.
Your task: produce a HIGHLY ACCURATE RepoBrief that could let someone draw an architecture diagram.

STRICT RULES:
- ALL fields REQUIRED with substantive content. Empty lists unacceptable when evidence exists.
"""

    user_prompt = """Repository: https://github.com/kbimsara/Z-ScoreUniFinder
Commit: 9dd068c0
GitHub metadata:
  - Primary language: JavaScript
  - Stars: 3
  - Description: ML-Final
  - Topics: 

=== DETECTED PROJECT TYPES (from file tree markers — treat as ground truth) ===
Node/JS, Python, Docker, Next.js

=== GITHUB REPOSITORY METADATA (verified) ===
  - forks: 4
  - size_kb: 645

There is a frontend in Next.js and backend in FastAPI.
"""
    llm = ChatNVIDIA(
        api_key=settings.nvidia_api_key,
        model=settings.repo_analizer_model,
        temperature=0.0,
    ).with_thinking_mode(enabled=True)
    
    try:
        res = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        print("RAW CONTENT:", res.content)
    except Exception as e:
        print("ERROR:", e)

if __name__ == "__main__":
    asyncio.run(test_llm())
