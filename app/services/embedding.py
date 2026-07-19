import asyncio
import httpx
import logging
from typing import List
from app.config import settings

logger = logging.getLogger(__name__)


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
        self.batch_size = settings.embed_batch_size
        
        # Token bucket for RPM limit
        self.rpm_limit = getattr(settings, 'embed_rpm_limit', 35)
        self.tokens = self.rpm_limit
        self.last_refill = asyncio.get_event_loop().time()
        # NVIDIA free tier limit resets per minute
        self.refill_rate_per_sec = self.rpm_limit / 60.0
        # Lock to prevent race conditions on token bucket
        self.lock = asyncio.Lock()

    async def _acquire_token(self):
        """Token bucket rate limiter to respect RPM limits strictly."""
        async with self.lock:
            while True:
                now = asyncio.get_event_loop().time()
                time_passed = now - self.last_refill
                
                # Refill tokens
                new_tokens = time_passed * self.refill_rate_per_sec
                if new_tokens > 0:
                    self.tokens = min(self.rpm_limit, self.tokens + new_tokens)
                    self.last_refill = now
                
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                
                # Need to wait for next token
                wait_time = (1 - self.tokens) / self.refill_rate_per_sec
                logger.debug(f"  [Embed] Rate limiter: waiting {wait_time:.2f}s for next token...")
                await asyncio.sleep(wait_time)

    async def embed_chunks(self, chunks: List[str], input_type: str = "passage") -> List[List[float]]:
        """
        Embed a list of text chunks sequentially with an RPM-aware token bucket rate limiter.
        input_type: "passage" for documents, "query" for search queries.

        Root cause of 503 on large repos: NVIDIA free tier has a strict 40 RPM limit.
        Solution: use a token bucket rate limiter to stay below RPM limit.
        """
        if not chunks:
            return []

        batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
        all_embeddings: List[List[float]] = []

        for batch_idx, batch in enumerate(batches):
            # Throttle: Wait for token to stay within NVIDIA RPM limit
            await self._acquire_token()

            embeddings = await self._embed_single_batch(batch, batch_idx, len(batches))
            all_embeddings.extend(embeddings)

        logger.info(f"  [Embed] {len(chunks)} chunks embedded in {len(batches)} batches")
        return all_embeddings

    async def _embed_single_batch(
        self, batch: List[str], batch_idx: int, total_batches: int
    ) -> List[List[float]]:
        """Embed one batch with exponential backoff retry on server/network errors."""
        payload = {
            "input": batch,
            "model": self.model,
            "encoding_format": "float",
            "input_type": "passage",
        }
        max_retries = 6
        for attempt in range(max_retries):
            try:
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
                        f"  [Embed] Batch {batch_idx+1}/{total_batches}: "
                        f"{len(batch)} chunks -> {len(embeddings)} embeddings"
                    )
                    return embeddings
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                is_rate_limit = (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code in [429, 503]
                )
                is_server_error = (
                    isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code in [500, 502, 504]
                )
                is_network_error = isinstance(e, httpx.RequestError)

                if attempt < max_retries - 1:
                    if is_rate_limit:
                        # Quota window exhaustion — wait for reset
                        wait_time = 65
                        logger.warning(
                            f"  [Embed] Batch {batch_idx+1} quota-limited (503/429). "
                            f"Waiting {wait_time}s for quota reset... (Attempt {attempt+1}/{max_retries})"
                        )
                    elif is_server_error or is_network_error:
                        wait_time = 3 * (2 ** attempt)  # 3s, 6s, 12s for transient errors
                        logger.warning(
                            f"  [Embed] Batch {batch_idx+1} server/network error ({repr(e)}). "
                            f"Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})"
                        )
                    else:
                        raise
                    
                    await asyncio.sleep(wait_time)
                else:
                    raise
        return []
