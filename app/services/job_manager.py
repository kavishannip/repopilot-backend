"""
DB-backed async job manager for RepoPilot pipeline.

State lives in the `analysis_jobs` PostgreSQL table, not in memory.
Jobs survive process restarts, idle spin-down, and redeploys on free-tier hosts.

Job lifecycle:
  create_job() -> update_phase() -> complete_job() | fail_job()

Consumers poll via get_job(). Old jobs cleaned up on each create.
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from sqlalchemy.future import select
from sqlalchemy import delete

from app.database import async_session
from app.models.schema import AnalysisJob

logger = logging.getLogger(__name__)

# Completed/failed jobs older than this are deleted on next create_job() call
JOB_TTL_HOURS = 24


class JobManager:
    async def create_job(self, repo_url: str) -> str:
        """
        Insert a new job row in 'pending' state. Returns job_id (UUID str).
        Also runs cleanup of stale old jobs.
        """
        async with async_session() as session:
            # Cleanup stale jobs first (piggyback on each new job creation)
            cutoff = datetime.utcnow() - timedelta(hours=JOB_TTL_HOURS)
            await session.execute(
                delete(AnalysisJob).where(
                    AnalysisJob.updated_at < cutoff,
                    AnalysisJob.status.in_(["completed", "failed"])
                )
            )

            job = AnalysisJob(repo_url=repo_url, status="pending", phase="queued")
            session.add(job)
            await session.commit()
            await session.refresh(job)
            job_id = str(job.id)

        logger.info(f"[Job {job_id[:8]}] created for {repo_url}")
        return job_id

    async def update_phase(self, job_id: str, phase: str) -> None:
        """Update job phase for progress tracking (running state implied)."""
        async with async_session() as session:
            stmt = select(AnalysisJob).where(AnalysisJob.id == uuid.UUID(job_id))
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job:
                job.status = "running"
                job.phase = phase
                job.updated_at = datetime.utcnow()
                await session.commit()
        logger.info(f"[Job {job_id[:8]}] phase -> {phase}")

    async def complete_job(self, job_id: str, result: Dict[str, Any]) -> None:
        """Mark job completed and store the RepoBrief JSON."""
        async with async_session() as session:
            stmt = select(AnalysisJob).where(AnalysisJob.id == uuid.UUID(job_id))
            res = await session.execute(stmt)
            job = res.scalar_one_or_none()
            if job:
                job.status = "completed"
                job.phase = "done"
                job.result = result
                job.updated_at = datetime.utcnow()
                await session.commit()
        logger.info(f"[Job {job_id[:8]}] completed")

    async def fail_job(self, job_id: str, error: str) -> None:
        """Mark job failed and store the error message."""
        async with async_session() as session:
            stmt = select(AnalysisJob).where(AnalysisJob.id == uuid.UUID(job_id))
            res = await session.execute(stmt)
            job = res.scalar_one_or_none()
            if job:
                job.status = "failed"
                job.phase = "failed"
                job.error = error
                job.updated_at = datetime.utcnow()
                await session.commit()
        logger.error(f"[Job {job_id[:8]}] failed: {error[:120]}")

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch job state for polling. Returns None if job not found.
        Returns dict with: status, phase, result, error, created_at, updated_at
        """
        try:
            job_uuid = uuid.UUID(job_id)
        except ValueError:
            return None

        async with async_session() as session:
            stmt = select(AnalysisJob).where(AnalysisJob.id == job_uuid)
            res = await session.execute(stmt)
            job = res.scalar_one_or_none()
            if not job:
                return None
            return {
                "job_id": str(job.id),
                "repo_url": job.repo_url,
                "status": job.status,
                "phase": job.phase,
                "result": job.result,
                "error": job.error,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "updated_at": job.updated_at.isoformat() if job.updated_at else None,
            }


# Singleton — reuse across requests
job_manager = JobManager()
