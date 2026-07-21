import requests
import json
import sys

# The URL of your local analyzer API
API_URL = "http://localhost:5000/api/analyzer/analyze"

def test_analyze(repo_url: str):
    print(f"Submitting {repo_url} for analysis... This may take a minute or two as the agent works.\n")
    
    try:
        response = requests.post(
            API_URL, 
            json={"repo_url": repo_url},
            timeout=300 # 5 minute timeout since LangChain can take a while
        )
        
        response.raise_for_status()
        data = response.json()
        
        print("✅ Analysis Complete!\n")
        
        print("=== GitHub Metadata ===")
        metadata = data["data"]["github_metadata"]
        print(json.dumps(metadata, indent=2))
        
        print("\n=== Generated Markdown Brief ===")
        print(data["data"]["repo_brief_markdown"])
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error during API request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Server response: {e.response.text}")

if __name__ == "__main__":
    test_repo = "https://github.com/kavishannip/Ai-Diagrams"
    if len(sys.argv) > 1:
        test_repo = sys.argv[1]
    
    test_analyze(test_repo)
