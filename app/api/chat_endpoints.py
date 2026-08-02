from fastapi import APIRouter, HTTPException, Header, Query, Depends
from pydantic import BaseModel
from typing import Optional, List
import uuid
import logging
from sqlalchemy.future import select

from app.database import async_session
from app.models.schema import Repository, ChatSession, ChatMessage
from app.services.chat_service import RepoChatService
from app.auth import verify_jwt

router = APIRouter()
chat_service = RepoChatService()
logger = logging.getLogger(__name__)


class CreateSessionRequest(BaseModel):
    repo_id: str
    title: Optional[str] = "Chat Session"
    client_id: Optional[str] = None


class CreateSessionResponse(BaseModel):
    session_id: str
    repo_id: str
    client_id: str
    title: str
    created_at: str


class SendMessageRequest(BaseModel):
    session_id: str
    question: Optional[str] = None
    message: Optional[str] = None
    client_id: Optional[str] = None


class SendMessageResponse(BaseModel):
    session_id: str
    answer: str
    sufficient: bool
    retrieved_chunks_count: int
    fallback_triggered: bool


class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    repo_id: str
    client_id: str
    title: str
    messages: List[ChatMessageResponse]


def get_effective_client_id(header_id: Optional[str], body_id: Optional[str], param_id: Optional[str] = None) -> str:
    client_id = header_id or body_id or param_id
    if not client_id:
        # Auto-generate a client ID if none provided
        client_id = f"client-{uuid.uuid4()}"
    return client_id


@router.post("/session", response_model=CreateSessionResponse)
async def create_chat_session(
    req: CreateSessionRequest,
    x_client_id: Optional[str] = Header(None, alias="X-Client-ID"),
    user: dict = Depends(verify_jwt),
):
    """
    Create a new chat session for a repository.
    Enforces mandatory client_id ownership (scoped to Asgardeo user).
    """
    effective_client_id = user.get("sub") or get_effective_client_id(x_client_id, req.client_id)

    try:
        repo_uuid = uuid.UUID(req.repo_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Repository ID format.")

    async with async_session() as session:
        stmt = select(Repository).where(Repository.id == repo_uuid)
        repo_obj = (await session.execute(stmt)).scalar_one_or_none()
        if not repo_obj:
            raise HTTPException(status_code=404, detail="Repository not found. Analyze repository first.")

        chat_sess = ChatSession(
            repository_id=repo_uuid,
            user_id=effective_client_id,
            title=req.title or "Chat Session",
        )
        session.add(chat_sess)
        await session.commit()
        await session.refresh(chat_sess)

        return CreateSessionResponse(
            session_id=str(chat_sess.id),
            repo_id=str(repo_obj.id),
            client_id=effective_client_id,
            title=chat_sess.title,
            created_at=chat_sess.created_at.isoformat(),
        )


@router.post("/message", response_model=SendMessageResponse)
async def send_chat_message(
    req: SendMessageRequest,
    x_client_id: Optional[str] = Header(None, alias="X-Client-ID"),
    user: dict = Depends(verify_jwt),
):
    """
    Send a question to a chat session.
    Runs query condensation, vector search, reasoning LLM synthesis, and 1-pass fallback file scan if needed.
    """
    effective_client_id = user.get("sub") or get_effective_client_id(x_client_id, req.client_id)
    user_query = req.question or req.message
    if not user_query:
        raise HTTPException(status_code=400, detail="Field 'question' or 'message' is required.")

    try:
        res = await chat_service.generate_chat_response(
            session_id=req.session_id,
            user_id=effective_client_id,
            question=user_query,
        )
        return SendMessageResponse(**res)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.exception(f"Chat processing failed for session {req.session_id}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {e}")


@router.get("/session/{session_id}", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: str,
    x_client_id: Optional[str] = Header(None, alias="X-Client-ID"),
    client_id: Optional[str] = Query(None),
    user: dict = Depends(verify_jwt),
):
    """
    Retrieve message history for a session. Validates client ownership.
    """
    effective_client_id = user.get("sub") or x_client_id or client_id
    if not effective_client_id:
        raise HTTPException(status_code=400, detail="Missing X-Client-ID header or client_id query param.")

    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Session ID format.")

    async with async_session() as session:
        stmt = select(ChatSession).where(ChatSession.id == session_uuid)
        sess_obj = (await session.execute(stmt)).scalar_one_or_none()
        if not sess_obj:
            raise HTTPException(status_code=404, detail="ChatSession not found.")

        if sess_obj.user_id != effective_client_id:
            raise HTTPException(status_code=403, detail="Access denied: session belongs to another client.")

        stmt_msgs = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_uuid)
            .order_by(ChatMessage.created_at.asc())
        )
        msgs = (await session.execute(stmt_msgs)).scalars().all()

        return SessionHistoryResponse(
            session_id=str(sess_obj.id),
            repo_id=str(sess_obj.repository_id),
            client_id=sess_obj.user_id,
            title=sess_obj.title,
            messages=[
                ChatMessageResponse(
                    id=str(m.id),
                    role=m.role,
                    content=m.content,
                    created_at=m.created_at.isoformat(),
                )
                for m in msgs
            ],
        )


@router.delete("/session/{session_id}")
async def delete_chat_session(
    session_id: str,
    x_client_id: Optional[str] = Header(None, alias="X-Client-ID"),
    client_id: Optional[str] = Query(None),
    user: dict = Depends(verify_jwt),
):
    """
    Delete a chat session and all its message history. Validates client ownership.
    """
    effective_client_id = user.get("sub") or x_client_id or client_id
    if not effective_client_id:
        raise HTTPException(status_code=400, detail="Missing X-Client-ID header or client_id query param.")

    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Session ID format.")

    async with async_session() as session:
        stmt = select(ChatSession).where(ChatSession.id == session_uuid)
        sess_obj = (await session.execute(stmt)).scalar_one_or_none()
        if not sess_obj:
            raise HTTPException(status_code=404, detail="ChatSession not found.")

        if sess_obj.user_id != effective_client_id:
            raise HTTPException(status_code=403, detail="Access denied: session belongs to another client.")

        await session.delete(sess_obj)
        await session.commit()

    return {"status": "deleted", "session_id": session_id}
