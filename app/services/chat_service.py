import asyncio
import json
import logging
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field

from sqlalchemy.future import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.database import async_session
from app.models.schema import Repository, RepoChunk, ChatSession, ChatMessage
from app.services.llm import get_fast_llm, get_reasoning_llm
from app.services.pipeline import ProcessingPipeline
from app.services.chunker import CodeChunker
from app.services.embedding import OpenAIEmbeddingClient
from app.services.github_client import GitHubClient

logger = logging.getLogger(__name__)


class ChatAnswerResponse(BaseModel):
    answer: str = Field(
        description=(
            "Extremely concise, accurate, and direct answer to the user's question based strictly on the repository context. "
            "When explaining logic or asked for code, include exact verbatim code snippets from the context formatted in standard markdown code blocks "
            "with language and filepath (e.g. ```python:app/main.py or ```python with // File: app/main.py). Never invent or hallucinate code."
        )
    )
    sufficient: bool = Field(
        description="True ONLY if the provided context is completely sufficient to answer accurately with 100% precision. False if key repository files or code context are missing."
    )


class RepoChatService:
    def __init__(self):
        self.pipeline = ProcessingPipeline()
        self.github = GitHubClient()
        self.chunker = CodeChunker()
        self.embedder = OpenAIEmbeddingClient()
        self._repo_locks: Dict[uuid.UUID, asyncio.Lock] = {}

    def _get_repo_lock(self, repo_id: uuid.UUID) -> asyncio.Lock:
        if repo_id not in self._repo_locks:
            self._repo_locks[repo_id] = asyncio.Lock()
        return self._repo_locks[repo_id]

    async def condense_query(self, history: List[ChatMessage], question: str) -> str:
        """
        Condense question + recent chat history into a standalone search query.
        Uses fast LLM.
        """
        if not history:
            return question.strip()

        recent_turns = history[-6:]
        formatted_history = "\n".join(
            f"{m.role.capitalize()}: {m.content}" for m in recent_turns
        )

        prompt = (
            "Given the following conversation history and a follow-up question, "
            "rephrase the follow-up question to be a complete standalone search query suitable for repository vector search.\n"
            "Do NOT answer the question. Only return the standalone query text.\n\n"
            f"Chat History:\n{formatted_history}\n\n"
            f"Follow-up Question: {question}\n\n"
            "Standalone Query:"
        )

        try:
            fast_llm = get_fast_llm()
            res = await fast_llm.ainvoke(prompt)
            query = res.content.strip() if hasattr(res, "content") else str(res).strip()
            logger.info(f"[Chat] Condensed query: '{question}' -> '{query}'")
            return query or question
        except Exception as e:
            logger.warning(f"[Chat] Condense query failed ({e}). Using raw question.")
            return question.strip()

    async def generate_chat_response(
        self,
        session_id: str,
        user_id: str,
        question: str,
    ) -> Dict[str, Any]:
        """
        Main 5-step Chat-with-Repo workflow:
        1. Validate session & check/sync repo commits (5-min TTL)
        2. Condense query (Fast LLM)
        3. Initial vector retrieval from centralized repo_chunks
        4. Answer synthesis Pass 1 (Reasoning LLM with structured output)
        5. Fallback expansion (if sufficient=false): Fast LLM candidate selection -> fetch -> chunk -> embed -> DB upsert -> Pass 2 retry
        """
        session_uuid = uuid.UUID(session_id)

        # 1. Fetch Session & Repository
        async with async_session() as session:
            stmt = select(ChatSession).where(ChatSession.id == session_uuid)
            chat_sess = (await session.execute(stmt)).scalar_one_or_none()
            if not chat_sess:
                raise ValueError(f"ChatSession '{session_id}' not found.")

            if chat_sess.user_id != user_id:
                raise PermissionError("Access denied: session belongs to another user.")

            repo_id = chat_sess.repository_id

            stmt_repo = select(Repository).where(Repository.id == repo_id)
            repo_obj = (await session.execute(stmt_repo)).scalar_one()
            repo_url = repo_obj.repo_url
            branch = repo_obj.default_branch or "main"

            # Fetch recent message history
            stmt_msgs = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_uuid)
                .order_by(ChatMessage.created_at.asc())
            )
            history = list((await session.execute(stmt_msgs)).scalars().all())

        # 1b. Check & Sync Commits (5-min TTL)
        try:
            await self.pipeline.check_and_sync_commit(repo_url, force=False)
        except Exception as e:
            logger.warning(f"[Chat] Commit sync check failed: {e}")

        # 2. Condense Query
        condensed_query = await self.condense_query(history, question)

        # 3. Vector Retrieval (Pass 1)
        retrieved_chunks = await self.pipeline.search_chunks(
            query=condensed_query,
            repo_url=repo_url,
            top_k=6,
        )

        # 4. Answer Synthesis (Pass 1)
        answer_res = await self._synthesize_answer(
            question=question,
            history=history,
            chunks=retrieved_chunks,
        )

        fallback_triggered = False

        # 5. Fallback Pass (if sufficient == False)
        if not answer_res.sufficient:
            logger.info(f"[Chat] Context insufficient for query '{condensed_query}'. Triggering fallback file scan...")
            fallback_triggered = True

            repo_lock = self._get_repo_lock(repo_id)
            async with repo_lock:
                # 5a. Get already processed file paths via indexed query
                processed_paths = await self.pipeline.get_processed_file_paths(repo_id)

                # 5b. Fetch tree and filter out processed paths
                try:
                    tree_data = await self.github.get_file_tree_structure(repo_url, branch)
                    all_file_items = tree_data.get("files", [])
                    unprocessed_files = [
                        f["path"] for f in all_file_items
                        if f.get("path") and f["path"] not in processed_paths
                    ]

                    if unprocessed_files:
                        # 5c. Ask Fast LLM to pick up to 5 candidate files
                        candidate_files = await self._select_candidate_files(
                            unprocessed_files=unprocessed_files,
                            query=condensed_query,
                        )

                        if candidate_files:
                            logger.info(f"[Chat] Fast LLM selected fallback candidates: {candidate_files}")
                            # 5d. Fetch, chunk, embed candidates
                            contents = await self.github.get_files_content_batch(
                                repo_url=repo_url,
                                paths=candidate_files,
                                branch=branch,
                            )

                            new_chunks = []
                            for path in candidate_files:
                                cnt = contents.get(path)
                                if cnt:
                                    try:
                                        new_chunks.extend(self.chunker.chunk_file(path, cnt))
                                    except Exception as e:
                                        logger.warning(f"Failed to chunk candidate {path}: {e}")

                            if new_chunks:
                                texts = [c["text"] for c in new_chunks]
                                embeddings = await self.embedder.embed_chunks(texts, input_type="passage")

                                async with async_session() as session:
                                    file_counters = {}
                                    chunk_values = []
                                    for i, c in enumerate(new_chunks):
                                        clean_text = c["text"].replace("\x00", "")
                                        clean_metadata = {
                                            k: v.replace("\x00", "") if isinstance(v, str) else v
                                            for k, v in c["metadata"].items()
                                        }
                                        fpath = c["metadata"].get("path", "")
                                        c_idx = file_counters.get(fpath, 0)
                                        file_counters[fpath] = c_idx + 1

                                        chunk_values.append({
                                            "id": uuid.uuid4(),
                                            "repository_id": repo_id,
                                            "file_path": fpath,
                                            "chunk_index": c_idx,
                                            "chunk_text": clean_text,
                                            "chunk_metadata": clean_metadata,
                                            "embedding": embeddings[i],
                                        })

                                    stmt = pg_insert(RepoChunk).values(chunk_values)
                                    stmt = stmt.on_conflict_do_nothing(constraint="uq_repo_chunk_file_pos")
                                    await session.execute(stmt)
                                    await session.commit()
                                    logger.info(f"[Chat] Successfully indexed {len(new_chunks)} fallback chunks to DB!")

                            # 5e. Re-retrieve vector search with newly indexed chunks
                            retrieved_chunks = await self.pipeline.search_chunks(
                                query=condensed_query,
                                repo_url=repo_url,
                                top_k=10,
                            )

                            # 5f. Retry Answer Synthesis (Pass 2)
                            answer_res = await self._synthesize_answer(
                                question=question,
                                history=history,
                                chunks=retrieved_chunks,
                                is_fallback_pass=True,
                            )
                except Exception as e:
                    logger.exception(f"[Chat] Fallback file scan encountered error: {e}")

            # If still insufficient after fallback round, return clear notice
            if not answer_res.sufficient:
                if not answer_res.answer.strip():
                    answer_res.answer = "Information not found in the repository."

        # Save conversation to DB
        async with async_session() as session:
            user_msg = ChatMessage(
                session_id=session_uuid,
                role="user",
                content=question,
            )
            assistant_msg = ChatMessage(
                session_id=session_uuid,
                role="assistant",
                content=answer_res.answer,
            )
            session.add_all([user_msg, assistant_msg])

            stmt_sess = select(ChatSession).where(ChatSession.id == session_uuid)
            sess_obj = (await session.execute(stmt_sess)).scalar_one()
            sess_obj.updated_at = datetime.utcnow()

            await session.commit()

        return {
            "session_id": session_id,
            "answer": answer_res.answer,
            "sufficient": answer_res.sufficient,
            "retrieved_chunks_count": len(retrieved_chunks),
            "fallback_triggered": fallback_triggered,
        }

    async def _synthesize_answer(
        self,
        question: str,
        history: List[ChatMessage],
        chunks: List[Dict[str, Any]],
        is_fallback_pass: bool = False,
    ) -> ChatAnswerResponse:
        """
        Synthesize final answer pass using Reasoning LLM with structured output.
        Enforces strict brevity, anti-hallucination, and precision.
        """
        context_str = ""
        for i, c in enumerate(chunks, 1):
            fpath = c.get("file_path") or c.get("metadata", {}).get("path", "unknown")
            context_str += f"--- Chunk {i} (File: {fpath}) ---\n{c['text']}\n\n"

        history_str = ""
        if history:
            history_str = "\n".join(f"{m.role.capitalize()}: {m.content}" for m in history[-6:])

        prompt = (
            "You are a hyper-precise AI software engineer answering questions about a codebase.\n\n"
            "STRICT RULES:\n"
            "1. CONCISE & STRAIGHT TO THE POINT: Provide the SHORTEST possible accurate answer. Do NOT use conversational fluff (e.g., 'Based on the code...', 'Sure!', 'Here is...'). Answer with immediate factual precision.\n"
            "2. VERBATIM CODE SNIPPETS: When explaining code logic, architecture, functions, or when requested by the user, ALWAYS include actual code snippets from the provided retrieved chunks.\n"
            "3. NO HALLUCINATION OF CODE: Only provide code examples that are EXACT, VERBATIM snippets from the actual repository code context provided. NEVER invent, rewrite, or hallucinate new code, functions, or parameters not present in the chunks.\n"
            "4. MANDATORY CODE FORMAT: Format every code snippet strictly using standard markdown code blocks specifying the language and exact relative file path. Format as either:\n"
            "   ```<language>:<filepath>\n"
            "   <exact code snippet from context>\n"
            "   ```\n"
            "   or\n"
            "   ```<language>\n"
            "   // File: <filepath>\n"
            "   <exact code snippet from context>\n"
            "   ```\n"
            "5. HONESTY ABOUT MISSING CONTEXT: If the retrieved code chunks do NOT contain enough information to answer with 100% confidence, set `sufficient: false`.\n"
            "6. IF NOT FOUND: If essential implementation details or requested code snippets are not in the provided chunks, state concisely what is missing.\n\n"
            f"Recent Chat History:\n{history_str or 'None'}\n\n"
            f"Retrieved Code Chunks:\n{context_str or 'No relevant code chunks found.'}\n\n"
            f"User Question: {question}"
        )

        reasoning_llm = get_reasoning_llm()

        try:
            structured_llm = reasoning_llm.with_structured_output(ChatAnswerResponse)
            res = await structured_llm.ainvoke(prompt)
            if isinstance(res, ChatAnswerResponse):
                return res
        except Exception as e:
            logger.warning(f"[Chat] Structured output pass failed ({e}). Falling back to raw JSON parsing.")

        # Fallback raw prompt invocation with JSON formatting instructions
        json_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT: Return your response strictly as valid JSON matching this schema:\n"
            '{"answer": "string", "sufficient": true|false}\n'
        )
        try:
            raw_res = await reasoning_llm.ainvoke(json_prompt)
            content = raw_res.content if hasattr(raw_res, "content") else str(raw_res)
            parsed = _extract_json_from_llm(content)
            if isinstance(parsed, dict):
                return ChatAnswerResponse(
                    answer=parsed.get("answer", content),
                    sufficient=bool(parsed.get("sufficient", True)),
                )
        except Exception as ex:
            logger.error(f"[Chat] Raw JSON fallback failed: {ex}")
            return ChatAnswerResponse(
                answer="Unable to parse answer response from reasoning model.",
                sufficient=True,
            )

    async def _select_candidate_files(
        self,
        unprocessed_files: List[str],
        query: str,
    ) -> List[str]:
        """
        Ask Fast LLM to select up to 5 candidate file paths from unprocessed file tree.
        Filters out non-code binary assets first.
        """
        if not unprocessed_files:
            return []

        # Filter out non-code binary/asset files
        ignored_extensions = (
            ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".woff", ".woff2",
            ".ttf", ".eot", ".mp4", ".webm", ".zip", ".tar", ".gz", ".pdf",
            ".exe", ".dll", ".so", ".dylib", ".lock", ".min.js", ".min.css", ".map"
        )
        code_files = [
            f for f in unprocessed_files
            if not f.lower().endswith(ignored_extensions)
        ]

        if not code_files:
            return []

        sample_files = code_files[:400]

        prompt = (
            "You are a repository file selection agent. Given a list of repository file paths and a search query, "
            "select up to 5 candidate file paths that are most likely to contain the implementation or answer for the query.\n\n"
            f"Search Query: {query}\n\n"
            f"Available File Paths:\n" + "\n".join(sample_files) + "\n\n"
            "Return ONLY a JSON array of up to 5 file paths, e.g. [\"src/auth.py\", \"config/settings.py\"]. "
            "If no files look relevant, return []."
        )

        try:
            fast_llm = get_fast_llm()
            res = await fast_llm.ainvoke(prompt)
            content = res.content if hasattr(res, "content") else str(res)

            try:
                candidates = _extract_json_from_llm(content)
            except Exception:
                # Fallback: scan for line-by-line file path matches in output text
                candidates = []
                cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                for line in cleaned.splitlines():
                    line_clean = line.strip().strip("-*`\"'").strip()
                    if line_clean in unprocessed_files:
                        candidates.append(line_clean)

            if isinstance(candidates, list):
                valid = [p for p in candidates if isinstance(p, str) and p in unprocessed_files]
                return valid[:5]
        except Exception as e:
            logger.warning(f"[Chat] Candidate file selection failed: {e}")

        return []


def _extract_json_from_llm(content: str) -> Any:
    """
    Robustly extract and parse JSON (dict or list) from LLM output.
    Handles thinking tags (<think>...</think>), markdown code blocks, preambles, and raw JSON.
    """
    if not content:
        raise ValueError("LLM output is empty.")

    # 1. Strip reasoning/thinking tags (e.g. <think>...</think>)
    cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = content.strip()

    # 2. Try markdown code block extraction ```json ... ``` or ``` ... ```
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned)
    if code_block_match:
        block_content = code_block_match.group(1).strip()
        try:
            return json.loads(block_content)
        except Exception:
            pass

    # 3. Try parsing the cleaned text directly
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 4. Search for outer JSON array [...] or outer JSON object {...}
    array_match = re.search(r'(\[\s*[\s\S]*?\s*\])', cleaned)
    if array_match:
        try:
            return json.loads(array_match.group(1))
        except Exception:
            pass

    object_match = re.search(r'(\{\s*[\s\S]*?\s*\})', cleaned)
    if object_match:
        try:
            return json.loads(object_match.group(1))
        except Exception:
            pass

    raise ValueError(f"No valid JSON array or object found in LLM response: {content[:100]!r}")


