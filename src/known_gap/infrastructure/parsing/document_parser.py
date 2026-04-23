import io
from pathlib import Path

import pypdf

from src.known_gap.domain.services.document_parser import DocumentParser
from src.known_gap.shared.exceptions.base import PermanentException


class DispatchingDocumentParser(DocumentParser):
    SUPPORTED_EXTENSIONS = frozenset({".pdf", ".md", ".txt"})

    def parse(self, filename: str, data: bytes) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf(data, filename)
        if suffix in {".md", ".txt"}:
            return self._parse_utf8(data, filename)
        raise PermanentException(
            message=f"Unsupported file type: {suffix}",
            error_code="UNSUPPORTED_FILE_TYPE",
            details={
                "filename": filename,
                "supported": sorted(self.SUPPORTED_EXTENSIONS),
            },
        )

    def _parse_pdf(self, data: bytes, filename: str) -> str:
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise PermanentException(
                message=f"Failed to parse PDF: {e}",
                error_code="PDF_PARSE_ERROR",
                details={"filename": filename, "error_type": type(e).__name__},
            ) from e

    def _parse_utf8(self, data: bytes, filename: str) -> str:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise PermanentException(
                message="File is not valid UTF-8",
                error_code="DECODE_ERROR",
                details={"filename": filename, "error_type": type(e).__name__},
            ) from e
