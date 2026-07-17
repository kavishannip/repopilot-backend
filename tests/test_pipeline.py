import asyncio
import os
import sys

# Ensure the root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.pipeline import ProcessingPipeline

async def test_pipeline():
    pipeline = ProcessingPipeline()
    test_repo = "https://github.com/octocat/Hello-World"
    
    print("\n--- 1. First Scan (Should embed and store) ---")
    res1 = await pipeline.process_repo(test_repo)
    print("Result 1:", res1)
    
    print("\n--- 2. Second Scan (Should hit cache) ---")
    res2 = await pipeline.process_repo(test_repo)
    print("Result 2:", res2)
    assert res2["status"] == "cached", "Did not hit cache on second run!"
    
    print("\n--- 3. Test Search ---")
    query = "How to define an API endpoint?"
    print(f"Query: '{query}'")
    search_res = await pipeline.search_chunks(query, test_repo, top_k=3)
    for i, r in enumerate(search_res):
        print(f"\nResult {i+1} (Distance: {r['distance']:.4f}):")
        print(f"Path: {r['metadata'].get('path')} | Type: {r['metadata'].get('type')}")
        # print snippet
        snippet = repr(r["text"][:150]) + ("..." if len(r["text"]) > 150 else "")
        print(f"Snippet: {snippet}")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
