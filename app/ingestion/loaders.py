"""Load supported files into LangChain documents without API dependencies."""

from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_EXTENSIONS = frozenset({".md", ".markdown", ".pdf", ".txt"})


class DocumentLoadError(ValueError):
    """Raised when a document cannot be loaded safely."""


class UnsupportedFileTypeError(DocumentLoadError):
    """Raised when a path does not have a supported file extension."""


def load_document(path: str | Path) -> list[Document]:
    """Load one PDF, Markdown, or text file into LangChain documents.

    Markdown and text files produce one document. PDFs produce one document per
    page so that page metadata remains available for later citations.
    """
    file_path = _validate_path(Path(path))
    extension = file_path.suffix.lower()
    document_id = _create_document_id(file_path)
    metadata = {
        "document_id": document_id,
        "filename": file_path.name,
        "source": _display_source(file_path),
        "file_type": _file_type(extension),
    }

    if extension == ".pdf":
        return _load_pdf(file_path, metadata)
    return [_load_text(file_path, metadata)]


def load_documents(paths: Iterable[str | Path]) -> list[Document]:
    """Load several supported files while preserving the supplied order."""
    documents: list[Document] = []
    for path in paths:
        documents.extend(load_document(path))
    return documents


def create_document_id(filename: str, content: bytes) -> str:
    """Create the stable identifier used for an uploaded document."""
    normalized_stem = "-".join(
        part
        for part in Path(filename).stem.lower().replace("_", "-").split("-")
        if part
    )
    readable_stem = normalized_stem or "document"
    digest = sha256()
    digest.update(Path(filename).name.lower().encode("utf-8"))
    digest.update(b"\0")
    digest.update(content)
    return f"{readable_stem}-{digest.hexdigest()[:16]}"


def _validate_path(path: Path) -> Path:
    """Resolve a path and ensure it identifies a supported regular file."""
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise DocumentLoadError(f"Document does not exist: {path}")
    if not resolved_path.is_file():
        raise DocumentLoadError(f"Document path is not a file: {path}")

    extension = resolved_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        extension_label = extension or "<none>"
        raise UnsupportedFileTypeError(
            f"Unsupported document type '{extension_label}' for {path}. "
            f"Supported extensions: {supported}."
        )
    return resolved_path


def _create_document_id(path: Path) -> str:
    """Build a readable, deterministic ID from the filename and file content."""
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(f"Could not read document: {path}") from exc
    return create_document_id(path.name, content)


def _display_source(path: Path) -> str:
    """Prefer a portable project-relative source path when possible."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _file_type(extension: str) -> str:
    """Normalize supported extensions into public metadata values."""
    if extension in {".md", ".markdown"}:
        return "markdown"
    if extension == ".txt":
        return "text"
    return "pdf"


def _load_text(path: Path, metadata: dict[str, str]) -> Document:
    """Read a UTF-8 Markdown or text file."""
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(f"Document is not valid UTF-8: {path}") from exc
    except OSError as exc:
        raise DocumentLoadError(f"Could not read document: {path}") from exc
    return Document(page_content=content, metadata=metadata)


def _load_pdf(path: Path, metadata: dict[str, str]) -> list[Document]:
    """Extract each PDF page and attach a one-based page number."""
    try:
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentLoadError(f"PDF is encrypted and cannot be read: {path}")

        documents = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_metadata = {**metadata, "page": page_number}
            documents.append(
                Document(
                    page_content=page.extract_text() or "",
                    metadata=page_metadata,
                )
            )
        return documents
    except DocumentLoadError:
        raise
    except (OSError, PdfReadError, ValueError) as exc:
        raise DocumentLoadError(f"Could not read PDF document: {path}") from exc
