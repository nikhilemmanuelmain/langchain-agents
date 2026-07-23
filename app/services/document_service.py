"""Persistent document storage, registry, and indexing service."""

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import RLock

from app.ingestion.loaders import (
    SUPPORTED_EXTENSIONS,
    DocumentLoadError,
    create_document_id,
    load_document,
)
from app.ingestion.splitter import split_documents
from app.retrieval.vector_store import DocumentVectorStore
from app.schemas.documents import DocumentRecord

SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class DocumentServiceError(RuntimeError):
    """Base error for managed-document operations."""


class DocumentNotFoundError(DocumentServiceError):
    """Raised when a managed document ID does not exist."""


class InvalidDocumentError(DocumentServiceError):
    """Raised when an upload does not meet input requirements."""


class DocumentIndexError(DocumentServiceError):
    """Raised when accepted content cannot be indexed."""


class DocumentServiceUnavailableError(DocumentServiceError):
    """Raised when vector storage cannot currently be used."""


class DocumentService:
    """Manage document files, durable metadata, and their vector chunks."""

    def __init__(
        self,
        *,
        documents_directory: str | Path,
        registry_path: str | Path,
        vector_store_provider: Callable[[], DocumentVectorStore],
        chunk_size: int,
        chunk_overlap: int,
        max_document_size_bytes: int,
    ) -> None:
        self._documents_directory = Path(documents_directory).resolve()
        self._registry_path = Path(registry_path).resolve()
        self._vector_store_provider = vector_store_provider
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_document_size_bytes = max_document_size_bytes
        self._lock = RLock()
        self._documents_directory.mkdir(parents=True, exist_ok=True)
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def max_document_size_bytes(self) -> int:
        """Return the configured upload limit for bounded multipart reads."""
        return self._max_document_size_bytes

    def list_documents(self) -> list[DocumentRecord]:
        """Return all managed documents sorted by filename and ID."""
        with self._lock:
            records = self._read_registry()
        return sorted(
            records.values(), key=lambda item: (item.filename, item.document_id)
        )

    def get_document(self, document_id: str) -> DocumentRecord:
        """Return one managed document or raise a clear not-found error."""
        with self._lock:
            record = self._read_registry().get(document_id)
        if record is None:
            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
        return record

    def upload_document(
        self, filename: str, content: bytes
    ) -> tuple[DocumentRecord, bool]:
        """Store and synchronously index an upload, deduplicating unchanged files."""
        safe_filename = sanitize_filename(filename)
        self._validate_upload(safe_filename, content)
        checksum = sha256(content).hexdigest()
        document_id = create_document_id(safe_filename, content)

        with self._lock:
            records = self._read_registry()
            duplicate = next(
                (
                    record
                    for record in records.values()
                    if record.filename == safe_filename and record.checksum == checksum
                ),
                None,
            )
            if duplicate is not None:
                return duplicate, False

            record = DocumentRecord(
                document_id=document_id,
                filename=safe_filename,
                checksum=checksum,
                status="pending",
            )
            document_path = self._document_path(record)
            document_path.parent.mkdir(parents=True, exist_ok=True)
            document_path.write_bytes(content)
            records[document_id] = record
            self._write_registry(records)

        return self._index_document(record), True

    def reindex_document(self, document_id: str) -> DocumentRecord:
        """Reload and replace all vector chunks for a managed document."""
        return self._index_document(self.get_document(document_id))

    def delete_document(self, document_id: str) -> None:
        """Delete vector chunks, stored content, and registry metadata."""
        record = self.get_document(document_id)
        try:
            self._vector_store_provider().delete_document(document_id)
        except Exception as exc:
            raise DocumentServiceUnavailableError(
                "Vector storage is currently unavailable."
            ) from exc

        with self._lock:
            records = self._read_registry()
            document_path = self._document_path(record)
            document_path.unlink(missing_ok=True)
            try:
                document_path.parent.rmdir()
            except OSError:
                pass
            records.pop(document_id, None)
            self._write_registry(records)

    def _index_document(self, record: DocumentRecord) -> DocumentRecord:
        processing = record.model_copy(
            update={"status": "processing", "error": None, "indexed_at": None}
        )
        self._save_record(processing)
        try:
            documents = load_document(self._document_path(processing))
            for document in documents:
                document.metadata["document_id"] = processing.document_id
                document.metadata["filename"] = processing.filename
            chunks = split_documents(
                documents,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            if not chunks:
                raise DocumentIndexError("Document does not contain indexable text.")
            vector_store = self._vector_store_provider()
            vector_store.add_chunks(chunks)
        except DocumentLoadError as exc:
            failed = processing.model_copy(
                update={"status": "failed", "error": "Document parsing failed."}
            )
            self._save_record(failed)
            raise DocumentIndexError("Document parsing failed.") from exc
        except DocumentIndexError as exc:
            failed = processing.model_copy(
                update={"status": "failed", "error": str(exc)}
            )
            self._save_record(failed)
            raise
        except Exception as exc:
            failed = processing.model_copy(
                update={"status": "failed", "error": "Document indexing failed."}
            )
            self._save_record(failed)
            raise DocumentServiceUnavailableError(
                "Document indexing is currently unavailable."
            ) from exc

        indexed = processing.model_copy(
            update={
                "status": "indexed",
                "chunk_count": len(chunks),
                "indexed_at": datetime.now(UTC),
                "error": None,
            }
        )
        self._save_record(indexed)
        return indexed

    def _validate_upload(self, filename: str, content: bytes) -> None:
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise InvalidDocumentError(
                f"Unsupported document type '{extension or '<none>'}'. "
                f"Supported extensions: {supported}."
            )
        if not content:
            raise InvalidDocumentError("Uploaded document must not be empty.")
        if len(content) > self._max_document_size_bytes:
            raise InvalidDocumentError(
                "Uploaded document exceeds the "
                f"{self._max_document_size_bytes} byte limit."
            )

    def _document_path(self, record: DocumentRecord) -> Path:
        return self._documents_directory / record.document_id / record.filename

    def _save_record(self, record: DocumentRecord) -> None:
        with self._lock:
            records = self._read_registry()
            records[record.document_id] = record
            self._write_registry(records)

    def _read_registry(self) -> dict[str, DocumentRecord]:
        if not self._registry_path.exists():
            return {}
        try:
            payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
            return {
                document_id: DocumentRecord.model_validate(value)
                for document_id, value in payload.items()
            }
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise DocumentServiceError("Document registry could not be read.") from exc

    def _write_registry(self, records: dict[str, DocumentRecord]) -> None:
        temporary_path = self._registry_path.with_suffix(
            f"{self._registry_path.suffix}.tmp"
        )
        payload = {
            document_id: record.model_dump(mode="json")
            for document_id, record in records.items()
        }
        try:
            temporary_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary_path.replace(self._registry_path)
        except OSError as exc:
            raise DocumentServiceError(
                "Document registry could not be written."
            ) from exc


def sanitize_filename(filename: str) -> str:
    """Return a traversal-safe filename containing conservative characters."""
    basename = Path(filename.replace("\\", "/")).name.strip()
    sanitized = SAFE_FILENAME_PATTERN.sub("_", basename).strip("._")
    if not sanitized or not Path(sanitized).suffix:
        raise InvalidDocumentError("A valid filename with an extension is required.")
    return sanitized
