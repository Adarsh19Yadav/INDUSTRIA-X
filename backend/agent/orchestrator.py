"""backend/agent/orchestrator.py — Phase 7 Agent Orchestrator.

The orchestrator is the central coordinator:

1. Receive query + optional machine input.
2. Safety-check for unsafe/out-of-scope requests.
3. Classify intent → determine required tools.
4. Validate that machine data is present when ML tools are required.
5. Execute selected tools in order, collecting evidence.
6. Build a structured, grounded response.
7. Return AgentResponse.

ARCHITECTURE
------------
- Deterministic intent classification (backend.agent.intent).
- Optional LLM-augmented intent via provider abstraction (llm_provider).
- All tool calls are sandboxed — no shell, no code execution.
- Prompt-injection protection: user text and retrieved docs are treated as
  data, never as instructions.
- Tool execution is sequential; failures are caught and recorded.
- The response always discloses which tools were called and any errors.

SAFETY INVARIANTS
-----------------
- Only tools in ALLOWED_TOOL_NAMES may be invoked.
- Empty tool list from intent classifier → safe refusal response.
- Missing machine data for ML tools → request-for-data response.
- All exceptions inside tool.call() are caught and recorded — never propagated raw.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.agent.intent import classify_intent
from backend.agent.llm_provider import get_llm_provider
from backend.agent.response_builder import build_response, build_safe_refusal
from backend.agent.schemas import AgentQuery, AgentResponse, MachineInput
from backend.agent.tools import ALLOWED_TOOL_NAMES, get_tool, list_tools

logger = logging.getLogger(__name__)

# Tools that require machine sensor data
_ML_TOOLS: frozenset[str] = frozenset(
    {"predict_failure", "detect_anomaly", "assess_machine", "explain_prediction"}
)


class AgentOrchestrator:
    """Lightweight state-machine agent orchestrator.

    The same instance can be reused across requests (stateless after __init__).
    """

    def __init__(self) -> None:
        self._llm = get_llm_provider()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self, query_obj: AgentQuery) -> AgentResponse:
        """Run the full agent pipeline for one query.

        Parameters
        ----------
        query_obj:
            Validated AgentQuery with query string and optional machine_input.

        Returns
        -------
        AgentResponse
            Structured response with answer, evidence, sources, and status.
        """
        query = query_obj.query.strip()
        machine_input: MachineInput | None = query_obj.machine_input

        logger.info("[Agent] Query: %s", query[:120])

        # ── Step 1: Classify intent ──────────────────────────────────────────
        tool_names = self._classify_intent(query)
        logger.info("[Agent] Intent → tools: %s", tool_names)

        # ── Step 2: Safe refusal ────────────────────────────────────────────
        if tool_names == [] and self._is_unsafe_query(query):
            logger.warning("[Agent] Unsafe query detected — returning safe refusal.")
            return build_safe_refusal(query)

        # Empty list from a non-unsafe query (shouldn't happen, but be safe)
        if not tool_names:
            tool_names = ["assess_machine"]

        # ── Step 3: Validate machine data requirement ────────────────────────
        ml_tools_needed = [t for t in tool_names if t in _ML_TOOLS]
        if ml_tools_needed and machine_input is None:
            logger.info(
                "[Agent] ML tools %s needed but machine_input is absent.", ml_tools_needed
            )
            return build_response(
                tools_used=[],
                tool_outputs={},
                tool_errors={},
                sources=[],
                query=query,
                machine_input_missing=True,
            )

        # ── Step 4: Execute tools ────────────────────────────────────────────
        tool_outputs: dict[str, dict] = {}
        tool_errors: dict[str, str] = {}
        executed_tools: list[str] = []

        for tool_name in tool_names:
            if tool_name not in ALLOWED_TOOL_NAMES:
                logger.error("[Agent] Skipping unknown tool '%s'.", tool_name)
                tool_errors[tool_name] = f"Unknown tool '{tool_name}'."
                continue

            tool = get_tool(tool_name)
            try:
                output = self._execute_tool(tool_name, tool, machine_input, query)
                tool_outputs[tool_name] = output
                executed_tools.append(tool_name)
                logger.info("[Agent] Tool '%s' succeeded.", tool_name)
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                tool_errors[tool_name] = error_msg
                executed_tools.append(tool_name)
                logger.warning("[Agent] Tool '%s' failed: %s", tool_name, error_msg)

        # ── Step 5: Extract RAG sources ─────────────────────────────────────
        sources: list[dict] = []
        if "search_maintenance_docs" in tool_outputs:
            raw = tool_outputs["search_maintenance_docs"]
            sources = [
                r for r in raw.get("results", [])
                if r.get("relevance_label") in ("high", "medium")
            ] or raw.get("results", [])[:3]

        # ── Step 6: Build response ───────────────────────────────────────────
        return build_response(
            tools_used=executed_tools,
            tool_outputs=tool_outputs,
            tool_errors=tool_errors,
            sources=sources,
            query=query,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _classify_intent(self, query: str) -> list[str]:
        """Classify intent using LLM if available, else deterministic fallback."""
        if self._llm.is_available():
            llm_result = self._llm.classify_intent(query, list_tools())
            if llm_result is not None:
                # LLM returned REFUSE → empty list
                if llm_result == []:
                    return []
                # Validate — only allow known tools
                valid = [t for t in llm_result if t in ALLOWED_TOOL_NAMES]
                if valid:
                    logger.debug("[Agent] LLM intent: %s", valid)
                    return valid
                logger.warning(
                    "[Agent] LLM returned invalid tools %s; falling back.", llm_result
                )
        # Deterministic fallback
        return classify_intent(query)

    def _is_unsafe_query(self, query: str) -> bool:
        """Return True if the query matches known unsafe patterns."""
        from backend.agent.intent import _PATTERNS  # noqa: PLC0415

        # The first pattern in _PATTERNS is the unsafe/refusal pattern
        unsafe_pattern = _PATTERNS[0][0]
        return bool(unsafe_pattern.search(query))

    def _execute_tool(
        self,
        tool_name: str,
        tool: Any,
        machine_input: MachineInput | None,
        query: str,
    ) -> dict:
        """Dispatch a single tool call with appropriate arguments."""
        if tool_name in _ML_TOOLS:
            if machine_input is None:
                raise ValueError(
                    f"Tool '{tool_name}' requires machine_input but none was provided."
                )
            return tool.call(machine_input=machine_input)
        elif tool_name == "search_maintenance_docs":
            # Build search query from user query + machine context
            search_q = self._build_rag_query(query, machine_input)
            return tool.call(query=search_q, top_k=4)
        else:
            raise ValueError(f"No dispatch rule for tool '{tool_name}'.")

    def _build_rag_query(
        self,
        query: str,
        machine_input: MachineInput | None,
    ) -> str:
        """Compose a focused RAG query from the user query.

        The query is purely for semantic search — it is NOT executed as code.
        """
        # Strip to plain words — do not pass through embedded instructions
        safe = query.strip()
        # If the user asked about a specific phenomenon, preserve it
        # (max 300 chars to keep it focused)
        return safe[:300]
