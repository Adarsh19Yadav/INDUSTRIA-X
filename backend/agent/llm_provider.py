"""backend/agent/llm_provider.py — LLM provider abstraction for Phase 7.

Provides a clean interface for optional LLM-augmented intent classification
and response synthesis.  If no LLM is configured (no credentials provided),
the system gracefully falls back to the deterministic orchestrator.

SUPPORTED PROVIDERS
-------------------
- ``watsonx``  : IBM watsonx.ai (ibm-watsonx-ai SDK required)
- ``openai``   : OpenAI-compatible API (openai SDK required)
- ``none``     : No LLM — purely deterministic fallback

SECURITY
--------
- API keys are NEVER hard-coded.
- All credentials are read from environment variables via the Settings class.
- Provider selection is read from ``LLM_PROVIDER`` env var.
- If credentials are absent or the SDK is unavailable, falls back silently.

USAGE IN THE ORCHESTRATOR
-------------------------
The orchestrator calls ``get_llm_provider()`` to obtain the active provider.
If ``provider.is_available()`` returns False, the orchestrator uses only the
deterministic intent classifier and response builder.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseLLMProvider(ABC):
    """Abstract LLM provider."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider is configured and ready."""

    @abstractmethod
    def classify_intent(self, query: str, tool_names: list[str]) -> list[str] | None:
        """Ask the LLM to select tools for *query*.

        Returns an ordered list of tool names, or None if unavailable/failed.
        The orchestrator falls back to deterministic intent on None.
        """

    @abstractmethod
    def synthesise_response(
        self,
        query: str,
        tool_outputs: dict[str, Any],
        sources: list[dict],
    ) -> str | None:
        """Ask the LLM to synthesise a natural-language answer.

        Returns a string answer, or None if unavailable/failed.
        The orchestrator falls back to the structured response builder on None.
        """


# ---------------------------------------------------------------------------
# No-op provider (deterministic fallback)
# ---------------------------------------------------------------------------


class NoLLMProvider(BaseLLMProvider):
    """Fallback provider when no LLM is configured or available."""

    def is_available(self) -> bool:
        return False

    def classify_intent(self, query: str, tool_names: list[str]) -> list[str] | None:
        return None

    def synthesise_response(
        self,
        query: str,
        tool_outputs: dict[str, Any],
        sources: list[dict],
    ) -> str | None:
        return None


# ---------------------------------------------------------------------------
# watsonx.ai provider
# ---------------------------------------------------------------------------

_WATSONX_INTENT_PROMPT = """You are the intent classifier for an industrial predictive-maintenance agent.
The available tools are: {tool_names}.

Tool descriptions:
- predict_failure: predict machine failure probability from sensor readings
- detect_anomaly: detect whether machine sensor readings are anomalous
- assess_machine: perform full risk assessment (failure + anomaly + risk score)
- explain_prediction: explain the failure prediction using SHAP attribution
- search_maintenance_docs: retrieve maintenance guidance from the knowledge base

User query: "{query}"

Reply with ONLY a comma-separated list of tool names to invoke, in order, from the available tools.
If the query is about machine control, code execution, or is unsafe, reply with: REFUSE
Examples:
- "What is the failure probability?" → predict_failure
- "Is this machine anomalous?" → detect_anomaly
- "Why is this machine high risk?" → assess_machine,explain_prediction
- "Why is this high risk and what maintenance?" → assess_machine,explain_prediction,search_maintenance_docs
- "Shut down the machine" → REFUSE
"""


class WatsonxProvider(BaseLLMProvider):
    """IBM watsonx.ai provider using ibm-watsonx-ai SDK."""

    def __init__(self, api_key: str, project_id: str, url: str, model_id: str) -> None:
        self._api_key = api_key
        self._project_id = project_id
        self._url = url
        self._model_id = model_id
        self._model = None

    def _get_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        try:
            from ibm_watsonx_ai import Credentials  # noqa: PLC0415
            from ibm_watsonx_ai.foundation_models import ModelInference  # noqa: PLC0415

            creds = Credentials(api_key=self._api_key, url=self._url)
            self._model = ModelInference(
                model_id=self._model_id,
                credentials=creds,
                project_id=self._project_id,
            )
            return self._model
        except Exception as exc:
            logger.warning("WatsonxProvider: failed to initialise model — %s", exc)
            return None

    def is_available(self) -> bool:
        return bool(self._api_key and self._project_id) and self._get_model() is not None

    def classify_intent(self, query: str, tool_names: list[str]) -> list[str] | None:
        model = self._get_model()
        if model is None:
            return None
        try:
            prompt = _WATSONX_INTENT_PROMPT.format(
                tool_names=", ".join(tool_names),
                query=query[:500],
            )
            response = model.generate_text(prompt=prompt)
            return _parse_tool_list(response, tool_names)
        except Exception as exc:
            logger.warning("WatsonxProvider.classify_intent failed: %s", exc)
            return None

    def synthesise_response(
        self,
        query: str,
        tool_outputs: dict[str, Any],
        sources: list[dict],
    ) -> str | None:
        # Use structured response builder — LLM synthesis is optional enhancement
        return None


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

_OPENAI_INTENT_SYSTEM = (
    "You are the intent classifier for an industrial predictive-maintenance agent. "
    "Reply with ONLY a comma-separated list of tool names from the available set, "
    "or the single word REFUSE if the request is unsafe or out of scope."
)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible provider."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client = None

    def _get_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # noqa: PLC0415

            self._client = OpenAI(api_key=self._api_key)
            return self._client
        except Exception as exc:
            logger.warning("OpenAIProvider: failed to initialise client — %s", exc)
            return None

    def is_available(self) -> bool:
        return bool(self._api_key) and self._get_client() is not None

    def classify_intent(self, query: str, tool_names: list[str]) -> list[str] | None:
        client = self._get_client()
        if client is None:
            return None
        try:
            user_msg = (
                f"Available tools: {', '.join(tool_names)}.\n"
                f"Query: {query[:500]}"
            )
            completion = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _OPENAI_INTENT_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=50,
                temperature=0,
            )
            raw = completion.choices[0].message.content or ""
            return _parse_tool_list(raw, tool_names)
        except Exception as exc:
            logger.warning("OpenAIProvider.classify_intent failed: %s", exc)
            return None

    def synthesise_response(
        self,
        query: str,
        tool_outputs: dict[str, Any],
        sources: list[dict],
    ) -> str | None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_tool_list(raw: str, allowed: list[str]) -> list[str] | None:
    """Parse LLM output into a validated tool list.

    Returns None if the LLM responded REFUSE or returned nothing parseable.
    """
    text = raw.strip().upper()
    if "REFUSE" in text:
        return []  # empty list signals safe refusal
    parts = [p.strip().lower() for p in raw.split(",")]
    valid = [p for p in parts if p in allowed]
    return valid if valid else None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_llm_provider() -> BaseLLMProvider:
    """Return the configured LLM provider, or ``NoLLMProvider`` as fallback.

    Reads configuration from the application Settings.  Returns gracefully
    if credentials are absent or the required SDK is not installed.
    """
    try:
        from backend.core.config import get_settings  # noqa: PLC0415

        s = get_settings()
        provider_name = (s.LLM_PROVIDER or "").lower()

        if provider_name == "watsonx" and s.WATSONX_API_KEY and s.WATSONX_PROJECT_ID:
            return WatsonxProvider(
                api_key=s.WATSONX_API_KEY,
                project_id=s.WATSONX_PROJECT_ID,
                url=s.WATSONX_URL,
                model_id=s.WATSONX_MODEL_ID,
            )

        if provider_name == "openai" and s.OPENAI_API_KEY:
            return OpenAIProvider(
                api_key=s.OPENAI_API_KEY,
                model=s.OPENAI_MODEL,
            )
    except Exception as exc:
        logger.warning("get_llm_provider: could not read settings — %s", exc)

    return NoLLMProvider()
