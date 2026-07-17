import httpx
from typing import List
from app.config import settings

class NVIDIAEmbeddingClient:
    def __init__(self):
        self.api_key = settings.nvidia_api_key
        self.url = "https://integrate.api.nvidia.com/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.model = "nvidia/nemotron-3-embed-1b"

    async def embed_chunks(self, chunks: List[str], input_type: str = "passage") -> List[List[float]]:
        """
        Embed a list of text chunks.
        input_type: "passage" for documents, "query" for search queries.
        """
        if not chunks:
            return []

        # NVIDIA API limits batch sizes, usually 50-100. We'll batch by 50.
        batch_size = 50
        all_embeddings = []

        async with httpx.AsyncClient() as client:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                payload = {
                    "input": batch,
                    "model": self.model,
                    "encoding_format": "float",
                    "input_type": input_type
                }
                
                resp = await client.post(self.url, headers=self.headers, json=payload, timeout=60.0)
                resp.raise_for_status()
                data = resp.json()
                
                for item in data.get("data", []):
                    all_embeddings.append(item["embedding"])
                    
        return all_embeddings
