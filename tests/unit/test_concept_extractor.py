import pytest

from src.known_gap.application.services.concept_extractor import ConceptExtractor
from src.known_gap.domain.services.llm_provider import LLMProvider
from src.known_gap.shared.exceptions.base import PermanentException


class _StubLLM(LLMProvider):
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple[str, str, int]] = []

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.calls.append((system, user, max_tokens))
        return self._response


class TestConceptExtractor:
    async def test_parses_valid_json_response(self) -> None:
        llm = _StubLLM(
            response=(
                '{"concepts": ['
                '{"name": "Recursion", "description": "A function calling itself.", '
                '"domain": "programming"},'
                '{"name": "Base case", "description": "Terminates recursion.", '
                '"domain": "programming"}'
                "]}"
            )
        )
        extractor = ConceptExtractor(llm)
        result = await extractor.extract("Some text about recursion.")

        assert len(result) == 2
        assert result[0].canonical_name == "recursion"
        assert result[0].display_name == "Recursion"
        assert result[0].domain == "programming"
        assert result[1].canonical_name == "base case"

    async def test_strips_markdown_code_fences(self) -> None:
        llm = _StubLLM(
            response='```json\n{"concepts": [{"name": "Loops", "description": "x"}]}\n```'
        )
        extractor = ConceptExtractor(llm)
        result = await extractor.extract("text")

        assert len(result) == 1
        assert result[0].canonical_name == "loops"

    async def test_raises_on_malformed_json(self) -> None:
        llm = _StubLLM(response="I think the concepts are loops and functions.")
        extractor = ConceptExtractor(llm)
        with pytest.raises(PermanentException) as exc_info:
            await extractor.extract("text")
        assert exc_info.value.error_code == "CONCEPT_EXTRACTION_PARSE_ERROR"

    async def test_handles_empty_concept_list(self) -> None:
        llm = _StubLLM(response='{"concepts": []}')
        extractor = ConceptExtractor(llm)
        assert await extractor.extract("text") == []

    async def test_raises_on_missing_concepts_key(self) -> None:
        llm = _StubLLM(response='{"items": [{"name": "foo"}]}')
        extractor = ConceptExtractor(llm)
        with pytest.raises(PermanentException):
            await extractor.extract("text")

    async def test_skips_malformed_concept_entries(self) -> None:
        llm = _StubLLM(
            response=(
                '{"concepts": ['
                '{"name": "Valid", "description": "ok"},'
                '"not an object",'
                '{"description": "missing name"},'
                '{"name": "   ", "description": "blank name"},'
                '{"name": "Also Valid"}'
                "]}"
            )
        )
        extractor = ConceptExtractor(llm)
        result = await extractor.extract("text")

        assert [c.canonical_name for c in result] == ["valid", "also valid"]

    async def test_null_domain_is_preserved_as_none(self) -> None:
        llm = _StubLLM(
            response='{"concepts": [{"name": "Widget", "description": "gadget", "domain": null}]}'
        )
        extractor = ConceptExtractor(llm)
        result = await extractor.extract("text")

        assert result[0].domain is None
