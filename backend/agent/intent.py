"""backend/agent/intent.py — Deterministic intent classification.

Maps a user query string to an ordered list of tool names that the
orchestrator should invoke.

This is a lightweight keyword/pattern approach — no external LLM required.
If an LLM provider is configured, the orchestrator may override this with a
model-generated plan.  The deterministic path ensures the agent still works
correctly when no LLM is available.

SAFETY
------
This module only selects from ALLOWED_TOOL_NAMES.
User-supplied text is never executed or interpreted as code.
"""

from __future__ import annotations

import re

from backend.agent.tools import ALLOWED_TOOL_NAMES

# ---------------------------------------------------------------------------
# Intent patterns — ordered by specificity (most specific first)
# ---------------------------------------------------------------------------

# Each entry: (compiled regex, ordered tool list)
# The first matching pattern wins.
_PATTERNS: list[tuple[re.Pattern, list[str]]] = [
    # Explicit machine-control / unsafe requests → no tools
    (
        re.compile(
            r"shut\s*down|power\s*off|turn\s*off|stop\s+machine|"
            r"change\s+.{0,20}(rpm|speed|torque|temperature)|"
            r"set\s+.{0,20}(rpm|speed|torque|temperature)|"
            r"modif(y|ies|ied)\s+.{0,20}(model|threshold|parameter)|"
            r"delete\s+.{0,20}(knowledge|database|collection|model)|"
            r"execute\s+.{0,30}(command|code|script|shell)|"
            r"run\s+.{0,20}(python|bash|shell|command|script)|"
            r"ignore\s+(all|previous|safety|instruction|rule)|"
            r"reveal\s+.{0,20}(secret|key|credential|token|password)",
            re.IGNORECASE,
        ),
        [],  # empty → orchestrator returns safe refusal
    ),
    # Full diagnosis: risk + explanation + maintenance guidance
    (
        re.compile(
            r"(why|reason|cause|explain).{0,80}(risk|fail|danger|problem|issue)"
            r".{0,80}(maintenance|repair|fix|action|recommend|consider|do)",
            re.IGNORECASE,
        ),
        ["assess_machine", "explain_prediction", "search_maintenance_docs"],
    ),
    # Full diagnosis (alternative word order)
    (
        re.compile(
            r"(maintenance|repair|fix|action|recommend).{0,80}"
            r"(why|reason|cause|explain).{0,80}(risk|fail|danger|problem)",
            re.IGNORECASE,
        ),
        ["assess_machine", "explain_prediction", "search_maintenance_docs"],
    ),
    # Diagnosis without explicit maintenance ask
    (
        re.compile(
            r"(why|reason|cause|explain).{0,80}(risk|fail|high.risk|danger|problem)",
            re.IGNORECASE,
        ),
        ["assess_machine", "explain_prediction"],
    ),
    # Comprehensive assessment + maintenance
    (
        re.compile(
            r"(overall|full|complete|comprehensive).{0,60}"
            r"(assess|evaluat|analys|diagnos).{0,60}(maintenance|repair|recommend)",
            re.IGNORECASE,
        ),
        ["assess_machine", "explain_prediction", "search_maintenance_docs"],
    ),
    # Comprehensive assessment only
    (
        re.compile(
            r"(overall|full|complete|comprehensive).{0,60}"
            r"(assess|evaluat|analys|diagnos)",
            re.IGNORECASE,
        ),
        ["assess_machine", "explain_prediction"],
    ),
    # Knowledge-base only: bearing/failure-mode specific questions
    (
        re.compile(
            r"(knowledge.base|documentation|manual|guidance).{0,60}"
            r"(bearing|gear|lubrication|vibration|wear|overheat|failure|fault)",
            re.IGNORECASE,
        ),
        ["search_maintenance_docs"],
    ),
    (
        re.compile(
            r"(bearing|gear|lubrication|vibration|wear|overheat|failure|fault)"
            r".{0,80}(knowledge.base|documentation|manual|guidance|say|contain|tell|about)",
            re.IGNORECASE,
        ),
        ["search_maintenance_docs"],
    ),
    # Maintenance recommendation without prediction context
    (
        re.compile(
            r"\b(recommend|suggest|advise|guidance|should\s+i|what\s+to\s+do|"
            r"next\s+step|action|inspect|schedule)\b.{0,80}"
            r"\b(maintenance|repair|fix|service|check|replace)\b",
            re.IGNORECASE,
        ),
        ["assess_machine", "search_maintenance_docs"],
    ),
    # SHAP / explanation only  (must come before generic risk/assess)
    (
        re.compile(
            r"\bshap\b|\bfeature.importan|\battribut"
            r"|\bwhich\s+feature|\bwhat\s+feature"
            r"|\bcontribut|\bdriving\b|\bdrives\b",
            re.IGNORECASE,
        ),
        ["explain_prediction"],
    ),
    # Anomaly only  (must come before generic risk/assess)
    (
        re.compile(
            r"\banomal|\bunusual\b|\boutlier\b|\babnormal\b"
            r"|\bdeviation\b|\birregular\b|\bstrange\b",
            re.IGNORECASE,
        ),
        ["detect_anomaly"],
    ),
    # Risk / overall assessment
    (
        re.compile(
            r"\b(risk|danger|critical|warning|status|assess|evaluat|diagnos)\b",
            re.IGNORECASE,
        ),
        ["assess_machine"],
    ),
    # Failure probability
    (
        re.compile(
            r"\b(fail(ure)?|predict|probabilit|likelihood|chance|will\s+it\s+fail|"
            r"failure\s+prob)\b",
            re.IGNORECASE,
        ),
        ["predict_failure"],
    ),
    # Maintenance knowledge — no ML context needed
    (
        re.compile(
            r"\b(maintenance|repair|service|bearing|gear|lubrication|"
            r"vibration|wear|overheat|fault|mode)\b",
            re.IGNORECASE,
        ),
        ["search_maintenance_docs"],
    ),
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SAFE_REFUSAL_SENTINEL = "__SAFE_REFUSAL__"


def classify_intent(query: str) -> list[str]:
    """Map *query* to an ordered list of tool names.

    Returns an empty list when the query matches a known unsafe pattern.
    Returns a list with at least one tool name otherwise.
    If no pattern matches, defaults to ``['assess_machine']`` when machine
    data is typically expected, or an empty list (graceful fallback).

    Parameters
    ----------
    query:
        Raw user query string.

    Returns
    -------
    list[str]
        Ordered list of tool names from ALLOWED_TOOL_NAMES, or [] for a
        safe-refusal case.
    """
    for pattern, tools in _PATTERNS:
        if pattern.search(query):
            # Safety check: all returned tools must be in the registry
            valid = [t for t in tools if t in ALLOWED_TOOL_NAMES]
            return valid

    # Default fallback — ask for a combined assessment
    return ["assess_machine"]
