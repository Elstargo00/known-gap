import json
from typing import Any

from src.known_gap.domain.models.concept import ConceptMention
from src.known_gap.domain.services.llm_provider import LLMProvider
from src.known_gap.shared.exceptions.base import PermanentException


class ConceptExtractor:
    SYSTEM_PROMPT = (
        "Extract the key concepts from the given text. A concept is a noun "
        "or short noun phrase that names an idea the reader must know in "
        "order to understand the text.\n\n"
        "Return ONLY a JSON object with this exact structure:\n"
        '{"concepts": [{"name": "<concept>", "description": "<one sentence>", '
        '"domain": "<field or null>"}]}\n\n'
        "Rules:\n"
        '- "name" is a short canonical noun phrase (2-4 words max).\n'
        '- "description" is a one-sentence gloss.\n'
        '- "domain" is the field it belongs to (e.g. "python", "biology", '
        '"machine learning") or null if unclear.\n'
        "- Return JSON only. No prose. No markdown fences."
    )

    def __init__(self, llm: LLMProvider, max_tokens: int = 1024) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    async def extract(self, text: str) -> list[ConceptMention]:
        raw = await self._llm.complete(
            system=self.SYSTEM_PROMPT,
            user=f"Text:\n{text}\n\nExtract concepts now.",
            max_tokens=self._max_tokens,
        )
        payload = self._parse(raw)
        return self._to_mentions(payload)

    def _parse(self, raw: str) -> dict[str, Any]:
        candidate = self._strip_fences(raw)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            raise PermanentException(
                message="Concept extractor received non-JSON response",
                error_code="CONCEPT_EXTRACTION_PARSE_ERROR",
                details={"raw_preview": raw[:500], "error": str(e)},
            ) from e
        if not isinstance(data, dict):
            raise PermanentException(
                message="Concept extractor response is not a JSON object",
                error_code="CONCEPT_EXTRACTION_PARSE_ERROR",
                details={"raw_preview": raw[:500]},
            )
        return data

    def _to_mentions(self, payload: dict[str, Any]) -> list[ConceptMention]:
        concepts = payload.get("concepts")
        if not isinstance(concepts, list):
            raise PermanentException(
                message="Concept extractor response missing 'concepts' list",
                error_code="CONCEPT_EXTRACTION_PARSE_ERROR",
                details={"payload_keys": list(payload.keys())},
            )

        mentions: list[ConceptMention] = []
        for item in concepts:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            display = name.strip()
            description = item.get("description") or ""
            domain = item.get("domain")
            mentions.append(
                ConceptMention(
                    canonical_name=display.lower(),
                    display_name=display,
                    description=description if isinstance(description, str) else "",
                    domain=domain if isinstance(domain, str) and domain.strip() else None,
                )
            )
        return mentions

    @staticmethod
    def _strip_fences(raw: str) -> str:
        text = raw.strip()
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
