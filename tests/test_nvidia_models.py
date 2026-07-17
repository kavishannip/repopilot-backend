import asyncio
import httpx
import os

async def list_models():
    api_key = os.environ.get("NVIDIA_API_KEY")
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("NVIDIA_API_KEY")
    url = "https://integrate.api.nvidia.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            for model in resp.json().get("data", []):
                print(model["id"])
        else:
            print("Failed", resp.status_code, resp.text)

if __name__ == "__main__":
    asyncio.run(list_models())
