import asyncio
import os
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

async def test_groq_embedding():
    client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
    try:
        print("Calling Groq embeddings API...")
        # attempt to use nomic-embed-text-v1_5
        # wait, the API might be client.embeddings.create(...)
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
