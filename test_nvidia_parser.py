import asyncio
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from app.config import settings
from app.models.brief import LLMRepoBrief
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

async def test_llm():
    parser = PydanticOutputParser(pydantic_object=LLMRepoBrief)
    format_instructions = parser.get_format_instructions()
    
    system_prompt = f"""You are an expert software architect producing a comprehensive repository brief.
Your task: produce a HIGHLY ACCURATE RepoBrief that could let someone draw an architecture diagram.

STRICT RULES:
- SYNTHESIZE CONFIDENTLY: The provided context contains everything you need. Synthesize the snippets, summaries, and structural maps to comprehensively fill out EVERY field. NEVER leave a field empty or output an empty list/object. If a specific detail is missing, infer the most likely architectural pattern based on standard practices for the detected frameworks.
- If highly uncertain about a specific detail, state your assumption in flagged_issues rather than leaving the field blank.
- ALL fields REQUIRED with substantive content. Empty lists unacceptable when evidence exists.

{format_instructions}
"""

    user_prompt = """Repository: https://github.com/kbimsara/Z-ScoreUniFinder
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
    
    try:
        res = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        print("--- RAW OUTPUT ---")
        try:
            print(res.content)
        except Exception as ex:
            print("Could not print raw content:", ex)
            
        print("--- PARSING ---")
        brief = parser.invoke(res)
        print("SUCCESSFULLY PARSED!")
        print(brief.model_dump_json(indent=2))
    except Exception as e:
        print("ERROR:", e)

if __name__ == "__main__":
    asyncio.run(test_llm())
