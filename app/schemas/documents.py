"""Public document-management API schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

DocumentStatus = Literal["pending", "processing", "indexed", "failed"]


class DocumentRecord(BaseModel):
    """Persisted public state for one managed document."""

    document_id: str
    filename: str
    checksum: str
    status: DocumentStatus
    chunk_count: int = 0
    indexed_at: datetime | None = None
    error: str | None = None
