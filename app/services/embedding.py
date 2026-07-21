import asyncio
import hashlib
import httpx
import logging
import random
from typing import List, Dict, Optional
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
        self.batch_size = getattr(settings, 'embed_batch_size', 20)
        
        # Token bucket for RPM limit
        self.rpm_limit = getattr(settings, 'embed_rpm_limit', 35)
        self.min_delay = getattr(settings, 'embed_min_delay_seconds', 2.0)
        
        self.tokens = self.rpm_limit
        self.last_refill = asyncio.get_event_loop().time()
        self.last_request_time = 0.0
        self.refill_rate_per_sec = self.rpm_limit / 60.0
        self.lock = asyncio.Lock()
        
        # In-memory chunk embedding cache (SHA-256 hash -> embedding vector)
        self._cache: Dict[str, List[float]] = {}

    def _hash_chunk(self, text: str, input_type: str) -> str:
        """Compute deterministic SHA-256 hash for chunk caching."""
        return hashlib.sha256(f"{input_type}:{text}".encode("utf-8")).hexdigest()

    async def _acquire_token(self):
        """
        Paced Token Bucket Rate Limiter.
        Enforces both overall RPM limit and a minimum inter-request delay (min_delay)
        to prevent instantaneous burst spikes (503/429) on NVIDIA NIM API gateway.
        """
        async with self.lock:
            while True:
                now = asyncio.get_event_loop().time()
                
                # 1. Pacing check: Ensure minimum delay since previous API call
                elapsed_since_last = now - self.last_request_time
                if elapsed_since_last < self.min_delay:
                    pacing_wait = self.min_delay - elapsed_since_last
                    logger.debug(f"  [Embed] Pacing: waiting {pacing_wait:.2f}s before next request...")
                    await asyncio.sleep(pacing_wait)
                    now = asyncio.get_event_loop().time()

                # 2. Token refill calculation
                time_passed = now - self.last_refill
                new_tokens = time_passed * self.refill_rate_per_sec
                if new_tokens > 0:
                    self.tokens = min(self.rpm_limit, self.tokens + new_tokens)
                    self.last_refill = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    self.last_request_time = asyncio.get_event_loop().time()
                    return

                # Need to wait for next token refill
                wait_time = (1 - self.tokens) / self.refill_rate_per_sec
                logger.debug(f"  [Embed] Rate limiter: waiting {wait_time:.2f}s for next token...")
                await asyncio.sleep(wait_time)

    async def embed_chunks(self, chunks: List[str], input_type: str = "passage") -> List[List[float]]:
        """
        Embed a list of text chunks with SHA-256 deduplication and paced batching.
        input_type: "passage" for documents, "query" for search queries.
        """
        if not chunks:
            return []

        # Step 1: Check cache for already embedded chunks
        chunk_hashes = [self._hash_chunk(c, input_type) for c in chunks]
        uncached_indices = [i for i, h in enumerate(chunk_hashes) if h not in self._cache]
        
        if not uncached_indices:
            logger.info(f"  [Embed] Cache hit for all {len(chunks)} chunks!")
            return [self._cache[h] for h in chunk_hashes]

        if len(uncached_indices) < len(chunks):
            logger.info(f"  [Embed] Cache hit for {len(chunks) - len(uncached_indices)}/{len(chunks)} chunks")

        uncached_chunks = [chunks[i] for i in uncached_indices]
        batches = [
            uncached_chunks[i:i + self.batch_size] 
            for i in range(0, len(uncached_chunks), self.batch_size)
        ]

        logger.info(
            f"  [Embed] Processing {len(uncached_chunks)} uncached chunks in {len(batches)} batches "
            f"(batch_size={self.batch_size}, pacing={self.min_delay}s)..."
        )

        # Step 2: Single persistent HTTP client for connection pooling across batches
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
        async with httpx.AsyncClient(limits=limits, timeout=60.0) as client:
            processed_count = 0
            batch_idx = 0
            while processed_count < len(uncached_chunks):
                current_batch_size = max(5, min(self.batch_size, getattr(settings, 'embed_batch_size', 15)))
                batch = uncached_chunks[processed_count : processed_count + current_batch_size]
                
                await self._acquire_token()
                batch_embeddings = await self._embed_batch_with_retry(
                    client=client,
                    batch=batch,
                    batch_idx=batch_idx,
                    total_batches=(len(uncached_chunks) + current_batch_size - 1) // current_batch_size,
                    input_type=input_type,
                )
                
                # Cache results
                for text, emb in zip(batch, batch_embeddings):
                    h = self._hash_chunk(text, input_type)
                    self._cache[h] = emb
                
                processed_count += len(batch)
                batch_idx += 1

        # Step 3: Reconstruct full output list in original order
        return [self._cache[h] for h in chunk_hashes]

    async def _embed_batch_with_retry(
        self,
        client: httpx.AsyncClient,
        batch: List[str],
        batch_idx: int,
        total_batches: int,
        input_type: str,
    ) -> List[List[float]]:
        """Embed one batch with Retry-After header parsing, exponential backoff, and adaptive splitting."""
        payload = {
            "input": batch,
            "model": self.model,
            "encoding_format": "float",
            "input_type": input_type,
        }
        max_retries = 6

        for attempt in range(max_retries):
            try:
                resp = await client.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = [item["embedding"] for item in data.get("data", [])]
                logger.debug(
                    f"  [Embed] Batch {batch_idx+1}/{total_batches}: "
                    f"{len(batch)} chunks -> {len(embeddings)} embeddings"
                )
                return embeddings

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code

                # Adaptive splitting if batch payload is rejected as too large (413 or 400 payload error)
                if status_code in (413, 400) and len(batch) > 1:
                    logger.warning(
                        f"  [Embed] Batch {batch_idx+1} payload size rejected ({status_code}). "
                        f"Splitting batch of {len(batch)} into 2 sub-batches..."
                    )
                    mid = len(batch) // 2
                    part1 = await self._embed_batch_with_retry(client, batch[:mid], batch_idx, total_batches, input_type)
                    await self._acquire_token()
                    part2 = await self._embed_batch_with_retry(client, batch[mid:], batch_idx, total_batches, input_type)
                    return part1 + part2

                is_rate_limit = status_code in (429, 503)
                is_server_error = status_code in (500, 502, 504)

                if attempt < max_retries - 1:
                    # Check for Retry-After header
                    retry_after_hdr = e.response.headers.get("Retry-After") or e.response.headers.get("retry-after")
                    if retry_after_hdr:
                        try:
                            wait_time = float(retry_after_hdr) + 1.0
                            logger.warning(
                                f"  [Embed] Batch {batch_idx+1} server specified Retry-After: {retry_after_hdr}s. "
                                f"Waiting {wait_time:.1f}s... (Attempt {attempt+1}/{max_retries})"
                            )
                        except ValueError:
                            wait_time = None
                    if wait_time is None:
                        if is_rate_limit:
                            # Dynamically step up inter-request pacing delay to alleviate API gateway congestion
                            self.min_delay = min(6.0, self.min_delay + 0.5)
                            self.batch_size = max(5, self.batch_size - 5)
                            logger.info(f"  [Embed] Dynamic Congestion Control: Increased pacing to {self.min_delay:.2f}s, reduced batch size to {self.batch_size}")
                            # Jittered exponential backoff for rate limits: ~4s, ~9s, ~19s, ~37s, ~65s max
                            base_backoff = min(65.0, (2 ** attempt) * 4.0)
                            jitter = random.uniform(0.5, 2.0)
                            wait_time = base_backoff + jitter
                            logger.warning(
                                f"  [Embed] Batch {batch_idx+1} quota/burst limited ({status_code}). "
                                f"Waiting {wait_time:.1f}s... (Attempt {attempt+1}/{max_retries})"
                            )
                        elif is_server_error:
                            wait_time = 3.0 * (2 ** attempt) + random.uniform(0.1, 1.0)
                            logger.warning(
                                f"  [Embed] Batch {batch_idx+1} server error ({status_code}). "
                                f"Retrying in {wait_time:.1f}s... (Attempt {attempt+1}/{max_retries})"
                            )
                        else:
                            raise

                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"  [Embed] Exhausted retries for batch {batch_idx+1}. "
                        f"Returning zero embeddings fallback."
                    )
                    return [[0.0] * 2048 for _ in batch]

            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    wait_time = 3.0 * (2 ** attempt) + random.uniform(0.1, 1.0)
                    logger.warning(
                        f"  [Embed] Batch {batch_idx+1} network error ({repr(e)}). "
                        f"Retrying in {wait_time:.1f}s... (Attempt {attempt+1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"  [Embed] Network error exhausted for batch {batch_idx+1}. Fallback to zero vectors.")
                    return [[0.0] * 2048 for _ in batch]

        return [[0.0] * 2048 for _ in batch]

