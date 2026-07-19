import asyncio
import os
from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from app.config import settings
from app.models.brief import LLMRepoBrief

async def test_llm():
    print("Model:", settings.repo_analizer_model)
    # Just a mini test
    llm = ChatNVIDIA(
        api_key=settings.nvidia_api_key,
        model=settings.repo_analizer_model,
        temperature=0.0,
    ).with_thinking_mode(enabled=True)
    
    llm_structured = llm.with_structured_output(LLMRepoBrief)
    
    prompt = "Analyze this repo: it is a python backend using fastapi. It has a main.py and requirements.txt."
    try:
        res = await llm_structured.ainvoke(prompt)
        print("RESULT:")
        print(res.model_dump_json(indent=2))
    except Exception as e:
        print("ERROR:", e)

if __name__ == "__main__":
    asyncio.run(test_llm())
