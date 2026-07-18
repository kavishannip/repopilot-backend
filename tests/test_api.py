import asyncio
import os
import sys
import httpx

# Ensure the root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from pprint import pprint

async def test_analyze_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        print("Testing POST /api/analyzer/analyze with invalid URL...")
        res1 = await client.post("/api/analyzer/analyze", json={"repo_url": "invalid-url"})
        print("Status:", res1.status_code)
        print("Response:", res1.json())
        assert res1.status_code == 400

        test_repo = "https://github.com/octocat/Hello-World"
        
        print(f"\nTesting POST /api/analyzer/analyze with {test_repo}...")
        res2 = await client.post("/api/analyzer/analyze", json={"repo_url": test_repo})
        print("Status:", res2.status_code)
        
        data = res2.json()
        print("\n--- Message ---")
        print(data.get("message"))
        
        print("\n--- Brief ---")
        if data.get("data"):
            pprint(data["data"])
        else:
            print("No brief data returned.")
            print(data)

        print(f"\nTesting POST /api/analyzer/analyze with {test_repo} (should hit cache)...")
        res3 = await client.post("/api/analyzer/analyze", json={"repo_url": test_repo})
        print("Status:", res3.status_code)
        print("Message:", res3.json().get("message"))
        assert res3.json().get("message") == "Returned from cache"

if __name__ == "__main__":
    asyncio.run(test_analyze_endpoint())
