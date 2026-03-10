"""Threads router — create, retrieve, and delete conversation threads.

Thread–agent binding:
    When a thread is created the chosen agent version is stored.  Every
    subsequent request on the same thread must use the same agent version —
    if they differ, HTTP 400 is returned.  This prevents subtle bugs where
    different agents share the same checkpoint state.
"""

import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status

from aug.api.schemas.threads import CreateThreadRequest, ThreadDetail, ThreadMetadata
from aug.api.security import require_api_key
from aug.core.graph import list_agents

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/threads",
    tags=["threads"],
    dependencies=[Depends(require_api_key)],
)


def _get_pool(request: Request):
    return request.app.state.db_pool


@router.post("", response_model=ThreadMetadata, status_code=status.HTTP_201_CREATED)
async def create_thread(body: CreateThreadRequest, request: Request) -> ThreadMetadata:
    """Create a new thread bound to *body.agent*."""
    if body.agent not in list_agents():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown agent '{body.agent}'. Available: {list_agents()}",
        )

    thread_id = str(uuid.uuid4())
    now = datetime.now(tz=UTC)
    pool = _get_pool(request)

    await pool.execute(
        """
        INSERT INTO threads (thread_id, agent_version, created_at, updated_at)
        VALUES ($1, $2, $3, $4)
        """,
        thread_id,
        body.agent,
        now,
        now,
    )
    logger.info("created thread=%s agent=%s", thread_id, body.agent)
    return ThreadMetadata(
        thread_id=thread_id,
        agent_version=body.agent,
        created_at=now,
        updated_at=now,
    )


@router.get("/{thread_id}", response_model=ThreadDetail)
async def get_thread(thread_id: str, request: Request) -> ThreadDetail:
    """Return thread metadata and message history."""
    pool = _get_pool(request)

    row = await pool.fetchrow(
        "SELECT thread_id, agent_version, created_at, updated_at FROM threads WHERE thread_id = $1",
        thread_id,
    )
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.")

    return ThreadDetail(
        thread_id=row["thread_id"],
        agent_version=row["agent_version"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        messages=[],  # TODO: hydrate from LangGraph checkpoint
    )


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(thread_id: str, request: Request) -> None:
    """Delete a thread and its associated checkpoint data."""
    pool = _get_pool(request)
    result = await pool.execute(
        "DELETE FROM threads WHERE thread_id = $1",
        thread_id,
    )
    if result == "DELETE 0":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.")
    logger.info("deleted thread=%s", thread_id)
