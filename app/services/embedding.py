import asyncio
import httpx
import logging
from typing import List
from app.config import settings

logger = logging.getLogger(__name__)

# NVIDIA embedding API limits batches to ~50 inputs
BATCH_SIZE = 50
# Max concurrent embedding API calls — don't hammer NVIDIA
EMBED_CONCURRENCY = 3


class NVIDIAEmbeddingClient:
    def __init__(self):
        self.api_key = settings.nvidia_api_key
        self.url = "https://integrate.api.nvidia.com/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.model = settings.nvidia_embed_model

    async def embed_chunks(self, chunks: List[str], input_type: str = "passage") -> List[List[float]]:
        """
        Embed a list of text chunks in parallel batches.
        input_type: "passage" for documents, "query" for search queries.

        Batches run concurrently (up to EMBED_CONCURRENCY at once) instead of
        sequentially — significantly faster for large repos with many chunks.
        """
        if not chunks:
            return []

        batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
        sem = asyncio.Semaphore(EMBED_CONCURRENCY)

        async def embed_batch(batch: List[str], batch_idx: int) -> List[List[float]]:
            async with sem:
                payload = {
                    "input": batch,
                    "model": self.model,
                    "encoding_format": "float",
                    "input_type": input_type,
                }
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        self.url,
                        headers=self.headers,
                        json=payload,
                        timeout=60.0,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    embeddings = [item["embedding"] for item in data.get("data", [])]
                    logger.debug(
                        f"  [Embed] Batch {batch_idx+1}/{len(batches)}: "
                        f"{len(batch)} chunks -> {len(embeddings)} embeddings"
                    )
                    return embeddings

        results = await asyncio.gather(*(embed_batch(b, i) for i, b in enumerate(batches)))

        # Flatten batch results preserving order
        all_embeddings = [emb for batch_result in results for emb in batch_result]
        logger.info(f"  [Embed] {len(chunks)} chunks embedded in {len(batches)} parallel batches")
        return all_embeddings
