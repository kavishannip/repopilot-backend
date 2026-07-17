import asyncio
import os
import httpx
from pprint import pprint

# Ensure the root directory is in python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings

async def test_nvidia_embedding():
    api_key = settings.nvidia_api_key
    if not api_key:
        print("Error: NVIDIA_API_KEY not set in .env")
        return

    url = "https://integrate.api.nvidia.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "input": ["def hello_world():\n    print('Hello world!')"],
        "model": "nvidia/nemotron-3-embed-1b",
        "encoding_format": "float",
        "input_type": "passage"
    }

    print("Calling NVIDIA Embeddings API...")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            print("Success!")
            embeddings = data.get("data", [])
            if embeddings:
                vec = embeddings[0]["embedding"]
                print(f"Dimension: {len(vec)}")
                print(f"Preview: {vec[:5]}")
            else:
                print("No embeddings returned:", data)
                
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e.response.status_code}")
            print(e.response.text)
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    asyncio.run(test_nvidia_embedding())
