import asyncio
import os
import sys

# Ensure the root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.llm import get_llm

async def test_llm():
    try:
        print("Initializing LLM...")
        llm = get_llm()
        
        print("Sending 'Hello, are you working?' to LLM...")
        response = await llm.ainvoke("Hello, are you working?")
        
        print("\n--- LLM Response ---")
        print(response.content)
        print("--------------------")
        print("LLM is working successfully!")
        
    except Exception as e:
        print(f"\nError testing LLM: {e}")
        print("Check if your model name in .env is supported by Groq.")

if __name__ == "__main__":
    asyncio.run(test_llm())
