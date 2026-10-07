"""backend/agent/response_builder.py — Assemble the final agent answer.

Takes structured tool outputs and produces a concise, evidence-grounded
decision-support response.

This module contains NO ML or LLM logic.  It purely formats structured
data into human-readable text and populates the AgentResponse model.

SAFETY
------
- Never claims guaranteed failure, prevention, or causal conclusions.
- Never generates maintenance recommendations without RAG evidence.
- Always includes appropriate limitation disclaimers.
- Treats all tool outputs as structured data — does not interpolate
  arbitrary text from retrieved documents into the answer without
  labelling them as retrieved content.
"""

from __future__ import annotations

from typing import Any

from backend.agent.schemas import AgentResponse, RAGSource, ToolEvidence


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_risk(risk_level: str, risk_score: float) -> str:
    emoji = {"CRITICAL": "🔴", "WARNING": "🟡", "NORMAL": "🟢"}.get(risk_level, "⚪")
    return f"{emoji} {risk_level} (score: {risk_score:.3f})"


def _top_shap_features(features: list[dict], n: int = 3) -> list[dict]:
    """Return the top-n SHAP contributors by magnitude."""
    return sorted(features, key=lambda f: f.get("contribution", 0), reverse=True)[:n]


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_assessment_section(assess: dict) -> str:
    lines = ["**Assessment**"]
    lines.append(
        f"- Risk level: {_fmt_risk(assess['risk_level'], assess['risk_score'])}"
    )
    lines.append(f"- Failure probability: {_fmt_pct(assess['failure_probability'])}")
    lines.append(
        f"- Anomaly status: {'⚠ Anomalous' if assess['is_anomaly'] else '✓ Normal'} "
        f"(severity index: {assess['anomaly_score']:.3f} — not a probability)"
    )
    return "\n".join(lines)


def _build_prediction_section(pred: dict) -> str:
    lines = ["**Failure Prediction**"]
    lines.append(f"- Failure probability: {_fmt_pct(pred['failure_probability'])}")
    lines.append(
        f"- Predicted class: {'Failure' if pred['predicted_failure'] else 'No failure'} "
        f"(threshold: {pred['threshold']:.3f})"
    )
    return "\n".join(lines)


def _build_anomaly_section(anom: dict) -> str:
    lines = ["**Anomaly Detection**"]
    lines.append(
        f"- Status: {'⚠ Anomalous' if anom['is_anomaly'] else '✓ Normal'}"
    )
    lines.append(
        f"- Anomaly severity index: {anom['anomaly_score']:.3f} "
        "(range 0–1, NOT a probability)"
    )
    return "\n".join(lines)


def _build_shap_section(explain: dict) -> str:
    lines = ["**Why (SHAP Feature Attribution)**"]
    top = _top_shap_features(explain.get("features", []))
    if not top:
        lines.append("- No SHAP feature data available.")
    else:
        for f in top:
            direction_label = {
                "increases_failure_risk": "↑ increases failure risk",
                "decreases_failure_risk": "↓ decreases failure risk",
            }.get(f.get("direction", ""), f.get("direction", ""))
            lines.append(
                f"- **{f['feature']}** = {f['value']}: {direction_label} "
                f"(SHAP contribution: {f['shap_value']:+.4f})"
            )
    lines.append(
        "\n*SHAP attribution reflects the model's feature weighting — "
        "it does NOT establish causation.*"
    )
    return "\n".join(lines)


def _build_maintenance_section(rag: dict) -> str:
    results = rag.get("results", [])
    lines = ["**Maintenance Knowledge**"]
    if not results:
        lines.append(
            "No sufficiently relevant maintenance knowledge-base evidence was retrieved. "
            "Do not act on unverified guidance."
        )
    else:
        high_medium = [r for r in results if r.get("relevance_label") in ("high", "medium")]
        shown = high_medium or results[:2]
        for r in shown[:3]:
            excerpt = r["content"][:300].replace("\n", " ")
            lines.append(
                f"- **Source:** `{r['source']}` (relevance: {r['relevance_label']})\n"
                f"  > {excerpt}…"
            )
    return "\n".join(lines)


def _build_recommendation(risk_level: str | None) -> str:
    if risk_level == "CRITICAL":
        return (
            "**Recommendation**\n"
            "This machine shows critical risk indicators. "
            "Prioritise inspection and review by qualified maintenance personnel "
            "before the next scheduled run."
        )
    elif risk_level == "WARNING":
        return (
            "**Recommendation**\n"
            "This machine shows elevated risk indicators. "
            "Schedule a maintenance review and monitor closely."
        )
    else:
        return (
            "**Recommendation**\n"
            "Risk indicators are within normal range. "
            "Continue regular monitoring and scheduled maintenance."
        )


_LIMITATIONS = (
    "**Limitations**\n"
    "- This is a decision-support prototype trained on the AI4I 2020 *synthetic* dataset.\n"
    "- Predictions have NOT been validated on real industrial equipment.\n"
    "- SHAP attribution does NOT establish causation.\n"
    "- Risk levels are NOT certified industrial safety classifications.\n"
    "- All consequential maintenance decisions must be reviewed by qualified personnel."
)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_response(
    tools_used: list[str],
    tool_outputs: dict[str, dict],
    tool_errors: dict[str, str],
    sources: list[dict],
    query: str,
    *,
    machine_input_missing: bool = False,
) -> AgentResponse:
    """Assemble the AgentResponse from tool outputs.

    Parameters
    ----------
    tools_used:
        Ordered list of tool names that were called.
    tool_outputs:
        Mapping from tool name → tool result dict.
    tool_errors:
        Mapping from tool name → error message string (for failed tools).
    sources:
        List of RAG RetrievalResult-like dicts.
    query:
        The original user query (for context).
    machine_input_missing:
        Set True when machine data was required but absent.
    """
    # ── Build evidence list ─────────────────────────────────────────────────
    evidence: list[ToolEvidence] = []
    for tool_name in tools_used:
        if tool_name in tool_outputs:
            evidence.append(ToolEvidence(tool=tool_name, result=tool_outputs[tool_name]))
        elif tool_name in tool_errors:
            evidence.append(
                ToolEvidence(tool=tool_name, result={"error": tool_errors[tool_name]})
            )

    # ── Build RAG sources ───────────────────────────────────────────────────
    rag_sources: list[RAGSource] = [
        RAGSource(
            source=s["source"],
            chunk_index=s["chunk_index"],
            distance=s["distance"],
            excerpt=s["content"][:300],
        )
        for s in sources
    ]

    # ── Handle missing machine input ────────────────────────────────────────
    if machine_input_missing:
        return AgentResponse(
            answer=(
                "To answer your question, I need the machine sensor readings. "
                "Please provide: Air temperature [K], Process temperature [K], "
                "Rotational speed [rpm], Torque [Nm], Tool wear [min], and Type (L/M/H)."
            ),
            tools_used=[],
            evidence=[],
            sources=[],
            status="error",
            error_detail="machine_input is required for this query but was not provided.",
        )

    # ── Determine overall status ────────────────────────────────────────────
    if tool_errors and not tool_outputs:
        status = "error"
    elif tool_errors:
        status = "partial"
    else:
        status = "success"

    # ── Build answer sections ────────────────────────────────────────────────
    sections: list[str] = []
    risk_level: str | None = None

    if "assess_machine" in tool_outputs:
        a = tool_outputs["assess_machine"]
        risk_level = a.get("risk_level")
        sections.append(_build_assessment_section(a))

    elif "predict_failure" in tool_outputs:
        sections.append(_build_prediction_section(tool_outputs["predict_failure"]))

    elif "detect_anomaly" in tool_outputs:
        sections.append(_build_anomaly_section(tool_outputs["detect_anomaly"]))

    if "explain_prediction" in tool_outputs:
        sections.append(_build_shap_section(tool_outputs["explain_prediction"]))

    if "search_maintenance_docs" in tool_outputs:
        sections.append(_build_maintenance_section(tool_outputs["search_maintenance_docs"]))

    if risk_level is not None:
        sections.append(_build_recommendation(risk_level))

    sections.append(_LIMITATIONS)

    # ── Error/partial note ──────────────────────────────────────────────────
    if tool_errors:
        error_names = ", ".join(sorted(tool_errors.keys()))
        sections.insert(
            0,
            f"⚠ The following tool(s) encountered errors: {error_names}. "
            "Results below reflect available data only.",
        )

    answer = "\n\n".join(sections) if sections else (
        "Unable to produce a response. Please check that machine data is provided "
        "and try again."
    )

    error_detail: str | None = None
    if tool_errors:
        error_detail = "; ".join(f"{k}: {v}" for k, v in tool_errors.items())

    return AgentResponse(
        answer=answer,
        tools_used=tools_used,
        evidence=evidence,
        sources=rag_sources,
        status=status,
        error_detail=error_detail,
    )


def build_safe_refusal(query: str) -> AgentResponse:
    """Return a safe refusal response for unsafe or out-of-scope queries."""
    return AgentResponse(
        answer=(
            "This request is outside the scope of the INDUSTRIA-X decision-support system.\n\n"
            "This agent is designed to:\n"
            "- Assess machine failure probability and risk level.\n"
            "- Explain prediction factors using SHAP attribution.\n"
            "- Retrieve maintenance guidance from the knowledge base.\n\n"
            "It does NOT:\n"
            "- Execute commands or code.\n"
            "- Control machine parameters.\n"
            "- Modify models or databases.\n"
            "- Provide guaranteed safety or maintenance outcomes.\n\n"
            "Please consult qualified maintenance personnel for operational decisions."
        ),
        tools_used=[],
        evidence=[],
        sources=[],
        status="error",
        error_detail="Query matched an unsafe or out-of-scope pattern. Request refused.",
    )
