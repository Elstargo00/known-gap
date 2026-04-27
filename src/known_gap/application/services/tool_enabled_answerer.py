"""LLM-driven answering with optional tool calls.

This service runs the Anthropic Messages tool-use loop directly so we
keep the existing `LLMProvider` semantics (transient/permanent
exception model, fallback chain) for the no-tool path. Tools are
provided as LangChain `BaseTool` instances — converted to Anthropic's
schema on the fly — so the tool surface is portable to any other
LangChain runtime.

If the primary tool-enabled call fails transiently, we fall back to a
plain text completion via the project's `LLMProvider` chain. The
fallback prompt tells the model it has no tools available, so it
answers purely from the system+user prompt.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock, ToolUseBlock
from langchain_core.tools import BaseTool

from src.known_gap.application.tools.graph_tools import (
    langchain_tools_to_anthropic_schema,
)
from src.known_gap.domain.services.llm_provider import LLMProvider
from src.known_gap.shared.exceptions.base import PermanentException, TransientException

logger = logging.getLogger(__name__)


class ToolEnabledAnswerer:
    """Runs `client.messages.create` with `tools=[...]` in a loop until
    the model stops asking for tool calls (or `max_iterations` is hit).

    Falls back to `fallback_llm.complete()` (no tools) on transient
    Anthropic failures."""

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        max_tokens: int,
        max_iterations: int,
        fallback_llm: LLMProvider,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._max_iterations = max(1, max_iterations)
        self._fallback = fallback_llm

    async def answer(
        self,
        system: str,
        user: str,
        tools: Sequence[BaseTool] | None = None,
    ) -> str:
        if not tools:
            return await self._fallback.complete(system, user, self._max_tokens)

        try:
            return await self._run_tool_loop(system, user, list(tools))
        except TransientException:
            logger.warning("tool-enabled answer failed transiently; falling back")
            return await self._fallback.complete(system, user, self._max_tokens)

    async def _run_tool_loop(
        self,
        system: str,
        user: str,
        tools: list[BaseTool],
    ) -> str:
        tool_schemas = langchain_tools_to_anthropic_schema(tools)
        tools_by_name = {t.name: t for t in tools}

        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]

        for _ in range(self._max_iterations):
            response = await self._invoke(system, messages, tool_schemas)

            if response.stop_reason != "tool_use":
                return self._collect_text(response.content)

            messages.append({"role": "assistant", "content": response.content})
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if not isinstance(block, ToolUseBlock):
                    continue
                tool_name = block.name
                tool_input = block.input or {}
                logger.info(
                    "tool call",
                    extra={"tool": tool_name, "args": tool_input},
                )
                output = await self._invoke_tool(tools_by_name, tool_name, tool_input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )

            if not tool_results:
                return self._collect_text(response.content)
            messages.append({"role": "user", "content": tool_results})

        # Hit iteration cap — make a final non-tool call to extract text.
        final = await self._invoke(system, messages, tool_schemas=None)
        return self._collect_text(final.content)

    async def _invoke(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, object]] | None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": messages,
        }
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        try:
            return await self._client.messages.create(**kwargs)
        except anthropic.APIConnectionError as e:
            raise self._transient(e) from e
        except anthropic.RateLimitError as e:
            raise self._transient(e) from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise self._transient(e) from e
            raise self._permanent(e) from e
        except anthropic.APIError as e:
            raise self._permanent(e) from e

    @staticmethod
    async def _invoke_tool(
        tools_by_name: dict[str, BaseTool],
        name: str,
        args: dict[str, Any],
    ) -> str:
        tool = tools_by_name.get(name)
        if tool is None:
            return json.dumps({"error": f"unknown tool {name!r}"})
        try:
            result = await tool.ainvoke(args)
        except Exception as e:  # noqa: BLE001 — errors must reach the model
            logger.exception("tool error", extra={"tool": name})
            return json.dumps({"error": f"tool execution failed: {e}"})
        return result if isinstance(result, str) else json.dumps(result)

    @staticmethod
    def _collect_text(blocks: Sequence[Any]) -> str:
        return "".join(block.text for block in blocks if isinstance(block, TextBlock)).strip()

    @staticmethod
    def _transient(error: Exception) -> TransientException:
        return TransientException(
            message=f"Anthropic tool-enabled call failed transiently: {error}",
            error_code="LLM_PROVIDER_ERROR",
            details={"provider": "anthropic", "error_type": type(error).__name__},
        )

    @staticmethod
    def _permanent(error: Exception) -> PermanentException:
        return PermanentException(
            message=f"Anthropic tool-enabled call failed: {error}",
            error_code="LLM_PROVIDER_ERROR",
            details={"provider": "anthropic", "error_type": type(error).__name__},
        )
