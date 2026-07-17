import asyncio
import os
import sys
from pprint import pprint

# Ensure the root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.github_client import GitHubClient, RepoSizeExceededError

async def test():
    client = GitHubClient()
    
    # Test 1: Real public repo (small-ish)
    test_repo = "https://github.com/tiangolo/fastapi"
    
    try:
        print(f"\n--- Fetch metadata for {test_repo} ---")
        meta = await client.get_repo_metadata(test_repo)
        pprint(meta)
        
        print(f"\n--- List files for {test_repo} ---")
        files = await client.list_files(test_repo)
        print(f"Total files returned (capped/filtered): {len(files)}")
        if files:
            print("Top 5 priority files:")
            for f in files[:5]:
                print(" -", f["path"])
                
        # Test 2: Fetch content of a specific file
        if files:
            target_file = files[0]["path"]
            print(f"\n--- Fetch content for {target_file} ---")
            content = await client.get_file_content(test_repo, target_file)
            print(f"Content length: {len(content) if content else 0} chars")
            print("Snippet:", repr(content[:100]) if content else "None")
            
    except RepoSizeExceededError as e:
        print(f"Size check worked: {e}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Large repo (e.g. linux kernel)
    large_repo = "https://github.com/torvalds/linux"
    print(f"\n--- Test Large Repo (should fail) {large_repo} ---")
    try:
        await client.get_repo_metadata(large_repo)
        print("FAIL: Did not block large repo.")
    except RepoSizeExceededError as e:
        print(f"PASS: Blocked large repo -> {e}")

if __name__ == "__main__":
    asyncio.run(test())
