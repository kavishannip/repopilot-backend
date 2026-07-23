import asyncio
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from app.config import settings
from app.models.brief import LLMRepoBrief
from langchain_core.messages import SystemMessage, HumanMessage

async def test_llm():
    system_prompt = """You are an expert software architect producing a comprehensive repository brief.
Your task: produce a HIGHLY ACCURATE RepoBrief that could let someone draw an architecture diagram.

STRICT RULES:
- SYNTHESIZE CONFIDENTLY: The provided context contains everything you need. Synthesize the snippets, summaries, and structural maps to comprehensively fill out EVERY field. NEVER leave a field empty or output an empty list/object. If a specific detail is missing, infer the most likely architectural pattern based on standard practices for the detected frameworks.
- If highly uncertain about a specific detail, state your assumption in flagged_issues rather than leaving the field blank.
- JSON SCHEMA COMPLIANCE: You must strictly adhere to the provided schema. Do NOT output a blank or empty JSON skeleton. EVERY string field must contain a detailed description (minimum 2 sentences). EVERY list field must contain all relevant items found in the context.
- NO MARKDOWN REPORTS: You must return the results SOLELY by populating the provided JSON/function schema. Do NOT write a markdown report or table. ALL information must be placed inside the schema fields.
"""

    user_prompt = """Repository: https://github.com/kavishannip/Ai-Diagrams
Commit: 9dd068c0
GitHub metadata:
  - Primary language: JavaScript
  - Stars: 3
  - Description: ML-Final

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
    
    llm_structured = llm.with_structured_output(LLMRepoBrief)
    
    try:
        res = await llm_structured.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        print(res.model_dump_json(indent=2))
    except Exception as e:
        print("ERROR:", e)

if __name__ == "__main__":
    asyncio.run(test_llm())
