import asyncio
import os
import sys

# Ensure the root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.summarizer import RepoSummarizer
from pprint import pprint

async def test_summarizer():
    summarizer = RepoSummarizer()
    test_repo = "https://github.com/octocat/Hello-World"
    
    print(f"\n--- Testing Summarizer on {test_repo} ---")
    try:
        brief = await summarizer.generate_brief(test_repo)
        print("\n--- Summary Result ---")
        # brief is a pydantic model, so we can call model_dump()
        pprint(brief.model_dump())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_summarizer())
