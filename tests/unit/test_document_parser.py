import pytest

from src.known_gap.infrastructure.parsing.document_parser import DispatchingDocumentParser
from src.known_gap.shared.exceptions.base import PermanentException


class TestDispatchingDocumentParser:
    def test_parses_txt(self) -> None:
        parser = DispatchingDocumentParser()
        assert parser.parse("note.txt", b"hello world") == "hello world"

    def test_parses_markdown(self) -> None:
        parser = DispatchingDocumentParser()
        result = parser.parse("doc.md", b"# heading\n\nbody")
        assert "# heading" in result
        assert "body" in result

    def test_rejects_unsupported_extension(self) -> None:
        parser = DispatchingDocumentParser()
        with pytest.raises(PermanentException) as excinfo:
            parser.parse("image.png", b"\x89PNG")
        assert excinfo.value.error_code == "UNSUPPORTED_FILE_TYPE"

    def test_is_case_insensitive_on_extension(self) -> None:
        parser = DispatchingDocumentParser()
        assert parser.parse("NOTE.TXT", b"hello") == "hello"

    def test_rejects_invalid_utf8(self) -> None:
        parser = DispatchingDocumentParser()
        with pytest.raises(PermanentException) as excinfo:
            parser.parse("bad.txt", b"\xff\xfe\x00\x00")
        assert excinfo.value.error_code == "DECODE_ERROR"
