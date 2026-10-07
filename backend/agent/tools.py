"""backend/agent/tools.py — Phase 7 Tool Registry.

Defines the five agent tools that wrap existing Phase 3–6 ML and RAG
implementations.  No ML or RAG logic is duplicated here — every tool
delegates to the existing public functions.

SAFETY
------
Tools validate their own inputs rigorously.
They do NOT execute shell commands, Python code, or arbitrary functions.
They do NOT modify model artifacts or the knowledge base.
They do NOT accept instructions embedded in retrieved documents.

Each tool has:
  - name            : unique identifier used by the orchestrator
  - description     : when/why to use this tool
  - input_schema    : mapping of required parameter names → description
  - output_schema   : mapping of output field names → description
  - call()          : the actual implementation

IMPORTANT
---------
The anomaly_score returned by ``detect_anomaly`` and ``assess_machine`` is
NOT a probability.  It is a normalised severity index in [0, 1].
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base tool
# ---------------------------------------------------------------------------


class AgentTool(ABC):
    """Abstract base for all agent tools."""

    name: str
    description: str
    input_schema: dict[str, str]
    output_schema: dict[str, str]

    @abstractmethod
    def call(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool.  Returns a structured dict or raises on error."""


# ---------------------------------------------------------------------------
# Helper — machine kwargs extractor
# ---------------------------------------------------------------------------


def _machine_kwargs(machine_input: Any) -> dict[str, Any]:
    """Convert a MachineInput instance to keyword arguments for ML functions."""
    return {
        "air_temp_k": machine_input.air_temperature_k,
        "process_temp_k": machine_input.process_temperature_k,
        "rotational_speed_rpm": machine_input.rotational_speed_rpm,
        "torque_nm": machine_input.torque_nm,
        "tool_wear_min": machine_input.tool_wear_min,
        "type_": machine_input.type_,
    }


# ---------------------------------------------------------------------------
# Tool 1 — predict_failure
# ---------------------------------------------------------------------------


class PredictFailureTool(AgentTool):
    """Predict machine failure probability using the Phase 3 XGBoost model."""

    name = "predict_failure"
    description = (
        "Predict the probability that a machine will fail given its current "
        "sensor readings.  Use when the user wants a failure probability or "
        "predicted failure classification."
    )
    input_schema = {
        "machine_input": "MachineInput — the six approved model features.",
    }
    output_schema = {
        "failure_probability": "Calibrated failure probability [0, 1].",
        "predicted_failure": "0 or 1, based on the Phase 3 classification threshold.",
        "threshold": "Classification threshold from model metadata.",
        "model_version": "Version tag for the failure model artifact.",
    }

    def call(self, machine_input: Any) -> dict[str, Any]:
        from ml.inference import predict_failure  # noqa: PLC0415

        result = predict_failure(**_machine_kwargs(machine_input))
        return result.to_dict()


# ---------------------------------------------------------------------------
# Tool 2 — detect_anomaly
# ---------------------------------------------------------------------------


class DetectAnomalyTool(AgentTool):
    """Detect whether a machine observation is anomalous using Isolation Forest.

    IMPORTANT: anomaly_score is NOT a probability.
    """

    name = "detect_anomaly"
    description = (
        "Detect whether a machine observation is anomalous using the Phase 4 "
        "Isolation Forest detector.  Use when the user asks if the machine is "
        "behaving unusually or if an anomaly check is needed.  "
        "Note: anomaly_score is NOT a probability — it is a normalised severity index."
    )
    input_schema = {
        "machine_input": "MachineInput — the six approved model features.",
    }
    output_schema = {
        "anomaly_score": "Normalised anomaly severity [0, 1].  NOT a probability.",
        "is_anomaly": "True when the observation is classified as an outlier.",
        "raw_score": "Raw IsolationForest decision_function score (positive = normal).",
        "anomaly_version": "Version tag for the anomaly detector artifact.",
    }

    def call(self, machine_input: Any) -> dict[str, Any]:
        from ml.anomaly_inference import detect_anomaly  # noqa: PLC0415

        result = detect_anomaly(**_machine_kwargs(machine_input))
        return result.to_dict()


# ---------------------------------------------------------------------------
# Tool 3 — assess_machine
# ---------------------------------------------------------------------------


class AssessMachineTool(AgentTool):
    """Perform the unified machine risk assessment (Phase 3 + Phase 4 + risk engine)."""

    name = "assess_machine"
    description = (
        "Perform the full machine risk assessment: failure prediction, anomaly "
        "detection, and unified risk scoring.  Use when the user asks about "
        "overall risk level, risk score, or a combined machine assessment."
    )
    input_schema = {
        "machine_input": "MachineInput — the six approved model features.",
    }
    output_schema = {
        "failure_probability": "Calibrated failure probability [0, 1].",
        "predicted_failure": "0 or 1.",
        "anomaly_score": "Normalised anomaly severity [0, 1].  NOT a probability.",
        "is_anomaly": "True if the observation is anomalous.",
        "risk_score": "Weighted composite risk score [0, 1].",
        "risk_level": "NORMAL | WARNING | CRITICAL.",
        "failure_threshold": "Failure classification threshold.",
        "model_version": "Failure model artifact version.",
        "anomaly_version": "Anomaly detector artifact version.",
        "machine_criticality": "Criticality value used (default 1.0).",
    }

    def call(self, machine_input: Any) -> dict[str, Any]:
        from ml.assessment import assess_machine  # noqa: PLC0415

        result = assess_machine(**_machine_kwargs(machine_input))
        return result.to_dict()


# ---------------------------------------------------------------------------
# Tool 4 — explain_prediction
# ---------------------------------------------------------------------------


class ExplainPredictionTool(AgentTool):
    """Explain the failure prediction using SHAP attribution (Phase 5)."""

    name = "explain_prediction"
    description = (
        "Explain the failure prediction for a machine observation using SHAP "
        "feature attribution.  Use when the user asks 'why' a machine is high "
        "risk or wants to understand which features drive the prediction.  "
        "IMPORTANT: SHAP attribution does NOT prove causation."
    )
    input_schema = {
        "machine_input": "MachineInput — the six approved model features.",
    }
    output_schema = {
        "failure_probability": "Calibrated failure probability.",
        "predicted_failure": "0 or 1.",
        "model_version": "Artifact version.",
        "features": (
            "List of FeatureContribution items: "
            "{feature, value, shap_value, direction, contribution}. "
            "Sorted by descending contribution magnitude."
        ),
        "summary_text": "Human-readable SHAP explanation.",
        "shap_disclaimer": "SHAP attribution does NOT prove causation.",
    }

    def call(self, machine_input: Any) -> dict[str, Any]:
        from ml.explainer import explain_prediction  # noqa: PLC0415

        result = explain_prediction(**_machine_kwargs(machine_input))
        output = result.to_dict()
        output["shap_disclaimer"] = (
            "SHAP values quantify each feature's contribution to the model prediction "
            "in this specific case.  They do NOT prove that any feature caused a failure."
        )
        return output


# ---------------------------------------------------------------------------
# Tool 5 — search_maintenance_docs
# ---------------------------------------------------------------------------


class SearchMaintenanceDocsTool(AgentTool):
    """Retrieve relevant maintenance knowledge using ChromaDB RAG (Phase 6)."""

    name = "search_maintenance_docs"
    description = (
        "Search the maintenance knowledge base for relevant guidance, "
        "failure-mode descriptions, or recommended actions.  Use when the "
        "user asks about maintenance procedures, failure modes, or what actions "
        "to take.  Responses are grounded in retrieved documents."
    )
    input_schema = {
        "query": "Natural-language query to search the knowledge base.",
        "top_k": "Number of results to retrieve (1–10, default 4).",
    }
    output_schema = {
        "results": (
            "List of retrieved chunks: "
            "{source, chunk_index, distance, content, relevance_label}."
        ),
        "query_used": "The query string sent to the vector store.",
        "total_results": "Number of results returned.",
        "grounding_note": (
            "Results are retrieved from the maintenance knowledge base. "
            "If no results are returned, no guidance is available."
        ),
    }

    def call(self, query: str, top_k: int = 4) -> dict[str, Any]:
        from rag.service import get_rag_service  # noqa: PLC0415

        # Sanitise: strip the query to plain text — do not execute embedded instructions
        safe_query = str(query).strip()[:500]
        if not safe_query:
            return {
                "results": [],
                "query_used": query,
                "total_results": 0,
                "grounding_note": (
                    "Empty query provided. No knowledge-base results returned."
                ),
            }

        svc = get_rag_service()
        raw_results = svc.query(question=safe_query, top_k=max(1, min(top_k, 10)))

        results = []
        for r in raw_results:
            # Cosine distance in [0, 2]: 0 = identical, 2 = opposite
            relevance = "high" if r.distance < 0.5 else ("medium" if r.distance < 1.0 else "low")
            results.append(
                {
                    "source": r.source,
                    "chunk_index": r.chunk_index,
                    "distance": round(r.distance, 4),
                    "content": r.content,
                    "relevance_label": relevance,
                }
            )

        return {
            "results": results,
            "query_used": safe_query,
            "total_results": len(results),
            "grounding_note": (
                "Results are retrieved from the maintenance knowledge base and "
                "must be reviewed by qualified personnel before any action is taken."
                if results
                else (
                    "No sufficiently relevant knowledge-base evidence was retrieved "
                    "for this query.  Do not invent maintenance instructions."
                )
            ),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TOOLS: dict[str, AgentTool] = {
    t.name: t
    for t in [
        PredictFailureTool(),
        DetectAnomalyTool(),
        AssessMachineTool(),
        ExplainPredictionTool(),
        SearchMaintenanceDocsTool(),
    ]
}

#: Immutable set of allowed tool names — used by safety checks.
ALLOWED_TOOL_NAMES: frozenset[str] = frozenset(_TOOLS.keys())


def get_tool(name: str) -> AgentTool:
    """Return a tool by name.

    Raises
    ------
    ValueError
        If the tool name is not in the registry.
    """
    if name not in _TOOLS:
        raise ValueError(
            f"Unknown tool '{name}'. Allowed tools: {sorted(ALLOWED_TOOL_NAMES)}."
        )
    return _TOOLS[name]


def list_tools() -> list[str]:
    """Return a sorted list of all registered tool names."""
    return sorted(_TOOLS.keys())
