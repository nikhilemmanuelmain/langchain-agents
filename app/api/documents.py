"""Managed-document upload, listing, deletion, and re-indexing routes."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from starlette.concurrency import run_in_threadpool

from app.dependencies import get_document_service
from app.schemas.documents import DocumentRecord
from app.services.document_service import (
    DocumentIndexError,
    DocumentNotFoundError,
    DocumentService,
    DocumentServiceError,
    DocumentServiceUnavailableError,
    InvalidDocumentError,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=DocumentRecord, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF, Markdown, or text document")],
    response: Response,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentRecord:
    """Store and index one uploaded documentation file."""
    try:
        content = await file.read(service.max_document_size_bytes + 1)
        record, created = await run_in_threadpool(
            service.upload_document,
            file.filename or "",
            content,
        )
    except InvalidDocumentError as exc:
        status_code = 413 if "exceeds" in str(exc) else 415
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except DocumentIndexError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DocumentServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DocumentServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not created:
        response.status_code = status.HTTP_200_OK
    return record


@router.get("", response_model=list[DocumentRecord])
def list_documents(
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> list[DocumentRecord]:
    """List public metadata for all managed documents."""
    return service.list_documents()


@router.get("/{document_id}", response_model=DocumentRecord)
def get_document(
    document_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentRecord:
    """Get one managed document record."""
    try:
        return service.get_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> None:
    """Delete a document and all associated vector chunks."""
    try:
        service.delete_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/{document_id}/reindex", response_model=DocumentRecord)
def reindex_document(
    document_id: str,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentRecord:
    """Reload and replace all indexed chunks for a document."""
    try:
        return service.reindex_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentIndexError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DocumentServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
