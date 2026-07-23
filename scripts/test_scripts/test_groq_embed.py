import asyncio
import os
import sys

# Ensure the root directory is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

async def test_groq_embedding():
    client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
    try:
        print("Calling Groq embeddings API...")
        response = await client.embeddings.create(
            input="Hello, world",
            model="nomic-embed-text-v1_5"
        )
        print("Success!")
        print(response.data[0].embedding[:5])
    except Exception as e:
        print("Error:", e)
        print("Attributes of client:")
        print(dir(client))

if __name__ == "__main__":
    asyncio.run(test_groq_embedding())
