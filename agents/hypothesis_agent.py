"""
hypothesis_agent.py - Ranks root cause hypotheses by causal probability.

The Hypothesis Agent:
1. Receives structured incident context from the Triage Agent
2. Queries the service dependency graph to understand blast radius
3. Applies causal reasoning: which services could CAUSE the observed pattern?
4. Produces ranked hypotheses with evidence chains and confidence scores

Key insight: upstream failures cause downstream 5xx/latency.
The Hypothesis Agent traces the causal path from symptoms to root cause.

Usage:
    from agents.hypothesis_agent import HypothesisAgent
    agent = HypothesisAgent()
    ranked = agent.rank(incident_context, service_graph)
"""

import os
import re
from dataclasses import dataclass

import anthropic
from loguru import logger

from core.incident_taxonomy import IncidentCategory


@dataclass
class RankedHypothesis:
    """A single ranked root cause hypothesis with evidence."""

    hypothesis: str
    confidence: float  # 0.0 - 1.0
    category: str  # IncidentCategory value
    evidence: list[str]
    ruling_out: str  # Single investigation step to confirm/deny
    upstream_service: str | None  # Which upstream service caused this?
    causal_chain: list[
        str
    ]  # e.g., ["redis_down", "cache_miss", "db_overload", "api_5xx"]


@dataclass
class HypothesisSet:
    """Complete set of ranked hypotheses for an incident."""

    top_hypothesis: RankedHypothesis
    alternatives: list[RankedHypothesis]
    sum_of_confidences: float  # Sum of raw confidence values; may exceed 1.0
    investigation_priority: str  # "database" | "network" | "deployment" | ...
    estimated_mttr_min: int


HYPOTHESIS_SYSTEM = """You are OncallCompass's Hypothesis Engine — a causal reasoning expert.

Given incident signals (alerts, metrics, logs) and service topology, rank root cause hypotheses
by causal probability.

Causal reasoning rules:
1. Latency spike WITHOUT CPU spike → suspect I/O: database, cache, or network
2. Error spike CORRELATED with deploy → suspect deployment regression (confidence 0.85+)
3. GRADUAL memory increase → suspect memory leak in application
4. SUDDEN OOM → suspect unbounded data loading (large query, file upload)
5. Errors on MULTIPLE services simultaneously → suspect shared dependency (database, cache, network)
6. Errors on ONE service only → suspect that service's code or its direct dependencies
7. Pattern: normal → degraded → worse over hours → suspect resource exhaustion (pool, disk, memory)
8. Pattern: normal → instant spike → suspect deployment or config change

For each hypothesis:
- confidence: 0.65+ for your top hypothesis (don't be uniformly uncertain)
- evidence: cite specific signals that support this hypothesis (e.g., "latency spike without CPU spike")
- ruling_out: ONE step that would confirm OR rule out this hypothesis in <5 minutes

Response format:
## Top Hypothesis
[Most likely root cause with confidence X.XX]

## Evidence
[Bullet points of supporting signals]

## Alternative Hypotheses
[Ranked list with confidence scores]

## Investigation Priority
[Which component to check first]"""


class HypothesisAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )

    def rank(
        self,
        incident: dict,
        service_graph: dict | None = None,
        max_hypotheses: int = 5,
    ) -> HypothesisSet:
        """
        Rank root cause hypotheses for an incident.

        Args:
            incident: Incident context dict with alerts, metrics, logs, context
            service_graph: Service dependency graph {service: [dependencies]}
            max_hypotheses: Maximum hypotheses to return

        Returns:
            HypothesisSet with ranked hypotheses
        """
        # Fast-path: rule-based pre-ranking for common patterns
        rule_hypotheses = self._apply_rules(incident)

        if rule_hypotheses:
            logger.info(
                f"Rule-based pre-ranking found {len(rule_hypotheses)} hypotheses"
            )

        prompt = self._build_prompt(incident, service_graph, rule_hypotheses)

        logger.info(
            f"Ranking hypotheses for incident with {len(incident.get('alerts', []))} alerts"
        )

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=HYPOTHESIS_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text

        hypotheses = self._parse_hypotheses(text, rule_hypotheses)
        return self._build_hypothesis_set(hypotheses)

    def _apply_rules(self, incident: dict) -> list[RankedHypothesis]:
        """Apply deterministic rules for common patterns before calling LLM."""
        hypotheses = []
        alerts = incident.get("alerts", [])
        metrics = incident.get("metrics", {})
        context = incident.get("context", {})

        alerts_text = " ".join(alerts).lower()
        recent_deploy = "deploy" in str(context.get("last_deploy", "")).lower()

        # Rule 1: Recent deployment + error spike = deployment regression
        if recent_deploy and any(
            kw in alerts_text for kw in ["error", "5xx", "failure"]
        ):
            hypotheses.append(
                RankedHypothesis(
                    hypothesis="Deployment regression — new code introduced a bug",
                    confidence=0.82,
                    category=IncidentCategory.DEPLOYMENT.value,
                    evidence=["Error spike correlated with recent deployment"],
                    ruling_out="Roll back deployment and check if errors resolve within 2 minutes",
                    upstream_service=None,
                    causal_chain=["deployment", "code_bug", "high_error_rate"],
                )
            )

        # Rule 2: Latency + no CPU spike = I/O bound
        cpu = metrics.get("cpu_pct", metrics.get("cpu", 0))
        latency = metrics.get("p99_ms", metrics.get("latency_p99", 0))
        if latency > 0 and cpu < 50 and latency > 500:
            hypotheses.append(
                RankedHypothesis(
                    hypothesis="Database slow query or connection pool exhaustion",
                    confidence=0.71,
                    category=IncidentCategory.DATABASE.value,
                    evidence=[
                        f"P99 latency {latency}ms with CPU only {cpu}% — I/O bound"
                    ],
                    ruling_out="Check pg_stat_activity or slow query log for queries > 1s",
                    upstream_service="database",
                    causal_chain=["slow_query", "high_latency"],
                )
            )

        # Rule 3: OOM alert
        if any(kw in alerts_text for kw in ["oom", "memory", "killed"]):
            hypotheses.append(
                RankedHypothesis(
                    hypothesis="Memory exhaustion — leak or unbounded data loading",
                    confidence=0.75,
                    category=IncidentCategory.MEMORY.value,
                    evidence=["OOM kill event detected"],
                    ruling_out="Check memory usage graph: gradual increase (leak) vs sudden spike (large load)?",
                    upstream_service=None,
                    causal_chain=["memory_exhaustion", "oom_kill", "pod_restart"],
                )
            )

        return hypotheses

    def _build_prompt(
        self,
        incident: dict,
        service_graph: dict | None,
        rule_hypotheses: list[RankedHypothesis],
    ) -> str:
        import json

        parts = []

        parts.append(
            "## Active Alerts\n"
            + "\n".join(f"- {a}" for a in incident.get("alerts", []))
        )

        if incident.get("metrics"):
            parts.append(f"## Metrics\n{json.dumps(incident['metrics'], indent=2)}")

        if incident.get("logs"):
            parts.append(f"## Log Snippet\n```\n{incident['logs'][:1000]}\n```")

        ctx = incident.get("context", {})
        if ctx:
            stack = ctx.get("stack", [])
            deploy = ctx.get("last_deploy", "unknown")
            parts.append(
                f"## Context\n- Stack: {', '.join(stack)}\n- Last deploy: {deploy}"
            )
            if ctx.get("recent_changes"):
                parts.append("- Recent changes: " + "; ".join(ctx["recent_changes"]))

        if service_graph:
            parts.append(f"## Service Graph\n{json.dumps(service_graph, indent=2)}")

        if rule_hypotheses:
            parts.append("## Pre-ranked Hypotheses (rule-based, verify or override)")
            for i, h in enumerate(rule_hypotheses):
                parts.append(f"{i + 1}. [{h.confidence:.0%}] {h.hypothesis}")

        parts.append(
            "\nRank root cause hypotheses by causal probability. "
            "Your top hypothesis must have confidence ≥ 0.65. "
            "Explain the evidence chain for each."
        )

        return "\n\n".join(parts)

    def _parse_hypotheses(
        self, text: str, rule_hypotheses: list[RankedHypothesis]
    ) -> list[RankedHypothesis]:
        """Parse LLM response into structured hypotheses."""
        hypotheses = list(rule_hypotheses)

        # Extract confidence values and hypothesis text from response
        pairs = re.findall(
            r"(?:hypothesis|cause)[:\s]+(.+?)(?:\n|$).*?confidence[:\s]+([0-9.]+)",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        for hypothesis_text, confidence_str in pairs[:5]:
            try:
                confidence = float(confidence_str)
                if confidence > 1.0:
                    confidence /= 100  # Handle "82" vs "0.82"

                # Check if already in rule-based set
                if not any(
                    h.hypothesis.lower()[:50] == hypothesis_text.strip().lower()[:50]
                    for h in hypotheses
                ):
                    cat = self._classify_category(hypothesis_text)
                    hypotheses.append(
                        RankedHypothesis(
                            hypothesis=hypothesis_text.strip()[:200],
                            confidence=confidence,
                            category=cat,
                            evidence=[],
                            ruling_out="",
                            upstream_service=None,
                            causal_chain=[],
                        )
                    )
            except (ValueError, IndexError):
                pass

        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses[:5]

    def _classify_category(self, text: str) -> str:
        """Classify hypothesis into incident category."""
        t = text.lower()
        if any(
            w in t for w in ["database", "postgres", "mysql", "query", "connection"]
        ):
            return IncidentCategory.DATABASE.value
        if any(w in t for w in ["deploy", "rollout", "release", "code", "bug"]):
            return IncidentCategory.DEPLOYMENT.value
        if any(w in t for w in ["memory", "oom", "heap", "leak"]):
            return IncidentCategory.MEMORY.value
        if any(w in t for w in ["network", "dns", "tls", "timeout", "partition"]):
            return IncidentCategory.NETWORK.value
        if any(w in t for w in ["cascade", "downstream", "upstream"]):
            return IncidentCategory.CASCADE.value
        return IncidentCategory.UNKNOWN.value

    def _build_hypothesis_set(
        self, hypotheses: list[RankedHypothesis]
    ) -> HypothesisSet:
        if not hypotheses:
            fallback = RankedHypothesis(
                hypothesis="Unknown root cause — investigate application logs",
                confidence=0.5,
                category=IncidentCategory.UNKNOWN.value,
                evidence=[],
                ruling_out="Check application error logs for stack traces",
                upstream_service=None,
                causal_chain=[],
            )
            hypotheses = [fallback]

        total_probability = sum(h.confidence for h in hypotheses)
        investigation_priority = hypotheses[0].category if hypotheses else "unknown"

        return HypothesisSet(
            top_hypothesis=hypotheses[0],
            alternatives=hypotheses[1:],
            sum_of_confidences=round(total_probability, 2),
            investigation_priority=investigation_priority,
            estimated_mttr_min=30,
        )
