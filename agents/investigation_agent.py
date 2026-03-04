"""
investigation_agent.py - Executes the investigation plan for an active incident.

The Investigation Agent:
1. Receives ranked hypotheses from the Hypothesis Agent
2. Executes investigation steps in priority order
3. Queries available data sources (logs, metrics, runbooks)
4. Confirms or rules out each hypothesis
5. Returns the confirmed root cause with evidence

This agent is the "hands" of OncallCompass — it follows the investigation plan
and updates confidence scores based on what it finds.

Usage:
    from agents.investigation_agent import InvestigationAgent
    agent = InvestigationAgent()
    findings = agent.investigate(hypothesis_set, incident_context)
"""

import os
from dataclasses import dataclass, field
from typing import Any

import anthropic
from loguru import logger

from agents.hypothesis_agent import HypothesisSet, RankedHypothesis
from core.incident_taxonomy import IncidentCategory
from core.runbook_engine import RunbookEngine


@dataclass
class InvestigationFinding:
    """Result of investigating a single hypothesis."""
    hypothesis: str
    status: str              # "confirmed" | "ruled_out" | "inconclusive"
    evidence_found: list[str]
    evidence_against: list[str]
    updated_confidence: float
    ruling_out_step: str  # Which step was used to investigate


@dataclass
class InvestigationResult:
    """Complete result of the investigation phase."""
    confirmed_root_cause: str | None
    confirmed_hypothesis: RankedHypothesis | None
    all_findings: list[InvestigationFinding]
    remaining_hypotheses: list[RankedHypothesis]
    investigation_steps_taken: list[str]
    context_gathered: dict   # Logs, metrics, etc. gathered during investigation
    confidence: float
    verdict: str             # "confirmed" | "narrowed" | "unknown"


INVESTIGATION_SYSTEM = """You are OncallCompass's Investigation Agent.

Given a ranked hypothesis and incident signals, determine if the hypothesis is
CONFIRMED, RULED OUT, or INCONCLUSIVE.

You have access to:
- Alert data (what's firing and at what values)
- Log snippets provided in the context
- Metric values at the time of incident
- Service topology and recent changes

For each hypothesis, reason through:
1. What evidence CONFIRMS this hypothesis?
2. What evidence RULES OUT this hypothesis?
3. What single check would definitively confirm/deny?

Respond in this format:
STATUS: [CONFIRMED|RULED_OUT|INCONCLUSIVE]
CONFIDENCE: [0.0-1.0]
EVIDENCE FOR: [bullet points]
EVIDENCE AGAINST: [bullet points]
NEXT CHECK: [single most important next step if INCONCLUSIVE]"""


class InvestigationAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        self.runbook_engine = RunbookEngine()

    def investigate(
        self,
        hypothesis_set: HypothesisSet,
        incident_context: dict,
        max_steps: int = 5,
    ) -> InvestigationResult:
        """
        Execute investigation plan to confirm or rule out hypotheses.

        Args:
            hypothesis_set: Ranked hypotheses from HypothesisAgent
            incident_context: Full incident context (alerts, logs, metrics, context)
            max_steps: Maximum investigation steps before returning

        Returns:
            InvestigationResult with confirmed root cause (if found)
        """
        all_hypotheses = [hypothesis_set.top_hypothesis] + hypothesis_set.alternatives
        # Track (hypothesis, raw_finding) pairs so we can always match them correctly.
        hyp_finding_pairs: list[tuple[RankedHypothesis, "_InvestigationFindingInternal"]] = []
        steps_taken = []
        context_gathered = {}
        steps_remaining = max_steps

        logger.info(f"Investigating {len(all_hypotheses)} hypotheses")

        for hypothesis in all_hypotheses:
            if steps_remaining <= 0:
                break

            # Check relevant runbook
            runbook_steps = self.runbook_engine.get_steps(
                alert_type=hypothesis.category,
                service_category=hypothesis.upstream_service or "general",
            )

            # Investigate this hypothesis
            finding = self._investigate_hypothesis(
                hypothesis, incident_context, runbook_steps
            )
            hyp_finding_pairs.append((hypothesis, finding))
            steps_remaining -= 1

            if finding.ruling_out_step:
                steps_taken.append(finding.ruling_out_step)

            if finding.status == "confirmed":
                logger.info(f"Root cause confirmed: {hypothesis.hypothesis}")
                public_findings = self._to_public_findings(hyp_finding_pairs)
                return InvestigationResult(
                    confirmed_root_cause=hypothesis.hypothesis,
                    confirmed_hypothesis=hypothesis,
                    all_findings=public_findings,
                    remaining_hypotheses=[h for h in all_hypotheses if h != hypothesis],
                    investigation_steps_taken=steps_taken,
                    context_gathered=context_gathered,
                    confidence=finding.updated_confidence,
                    verdict="confirmed",
                )

        # No single hypothesis confirmed — return best guess.
        public_findings = self._to_public_findings(hyp_finding_pairs)
        if hyp_finding_pairs:
            best_hyp, best_finding = max(
                hyp_finding_pairs, key=lambda pair: pair[1].updated_confidence
            )
            # Use the updated confidence from investigation, not the prior.
            best_confidence = best_finding.updated_confidence
        else:
            best_hyp = hypothesis_set.top_hypothesis
            best_confidence = hypothesis_set.top_hypothesis.confidence

        verdict = "narrowed" if hyp_finding_pairs else "unknown"

        return InvestigationResult(
            confirmed_root_cause=best_hyp.hypothesis if best_hyp else None,
            confirmed_hypothesis=best_hyp,
            all_findings=public_findings,
            remaining_hypotheses=all_hypotheses[1:],
            investigation_steps_taken=steps_taken,
            context_gathered=context_gathered,
            confidence=best_confidence,
            verdict=verdict,
        )

    def _investigate_hypothesis(
        self,
        hypothesis: RankedHypothesis,
        incident_context: dict,
        runbook_steps: list[str],
    ) -> "_InvestigationFindingInternal":
        """Use LLM to investigate a single hypothesis."""
        prompt = self._build_investigation_prompt(hypothesis, incident_context, runbook_steps)

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=INVESTIGATION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        return self._parse_investigation_result(text, hypothesis)

    def _build_investigation_prompt(
        self, hypothesis: RankedHypothesis, incident_context: dict, runbook_steps: list[str]
    ) -> str:
        import json
        parts = [
            f"## Hypothesis to Investigate\n{hypothesis.hypothesis}\n"
            f"Prior confidence: {hypothesis.confidence:.0%}"
        ]

        parts.append(f"## Current Evidence\n" + "\n".join(f"- {e}" for e in hypothesis.evidence))

        if hypothesis.ruling_out:
            parts.append(f"## Proposed Ruling-Out Step\n{hypothesis.ruling_out}")

        parts.append(f"## Alert Signals\n" + "\n".join(f"- {a}" for a in incident_context.get("alerts", [])))

        if incident_context.get("metrics"):
            parts.append(f"## Metrics\n{json.dumps(incident_context['metrics'], indent=2)}")

        if incident_context.get("logs"):
            parts.append(f"## Logs\n```\n{incident_context['logs'][:800]}\n```")

        if runbook_steps:
            parts.append("## Relevant Runbook Steps\n" + "\n".join(f"- {s}" for s in runbook_steps[:4]))

        parts.append("Determine if this hypothesis is CONFIRMED, RULED_OUT, or INCONCLUSIVE.")
        return "\n\n".join(parts)

    def _parse_investigation_result(self, text: str, hypothesis: RankedHypothesis) -> "_InvestigationFindingInternal":
        import re

        # Parse STATUS
        status_m = re.search(r"STATUS:\s*(CONFIRMED|RULED_OUT|INCONCLUSIVE)", text, re.IGNORECASE)
        status_raw = status_m.group(1).upper() if status_m else "INCONCLUSIVE"
        status_map = {"CONFIRMED": "confirmed", "RULED_OUT": "ruled_out", "INCONCLUSIVE": "inconclusive"}
        status = status_map.get(status_raw, "inconclusive")

        # Parse CONFIDENCE
        conf_m = re.search(r"CONFIDENCE:\s*([0-9.]+)", text, re.IGNORECASE)
        confidence = float(conf_m.group(1)) if conf_m else hypothesis.confidence
        if confidence > 1.0:
            confidence /= 100

        # Parse evidence lists
        for_m = re.search(r"EVIDENCE FOR:\s*(.*?)(?=EVIDENCE AGAINST:|NEXT CHECK:|$)", text, re.IGNORECASE | re.DOTALL)
        against_m = re.search(r"EVIDENCE AGAINST:\s*(.*?)(?=NEXT CHECK:|$)", text, re.IGNORECASE | re.DOTALL)
        next_m = re.search(r"NEXT CHECK:\s*(.*?)$", text, re.IGNORECASE | re.DOTALL)

        evidence_for = re.findall(r"[-•]\s+(.+)", for_m.group(1)) if for_m else []
        evidence_against = re.findall(r"[-•]\s+(.+)", against_m.group(1)) if against_m else []
        next_check = next_m.group(1).strip()[:200] if next_m else ""

        return _InvestigationFindingInternal(
            status=status,
            updated_confidence=confidence,
            evidence_for=evidence_for,
            evidence_against=evidence_against,
            ruling_out_step=next_check,
        )

    @staticmethod
    def _to_public_findings(
        pairs: "list[tuple[RankedHypothesis, _InvestigationFindingInternal]]",
    ) -> "list[InvestigationFinding]":
        """Convert internal findings to the public InvestigationFinding type."""
        return [
            InvestigationFinding(
                hypothesis=hyp.hypothesis,
                status=internal.status,
                evidence_found=internal.evidence_for,
                evidence_against=internal.evidence_against,
                updated_confidence=internal.updated_confidence,
                ruling_out_step=internal.ruling_out_step,
            )
            for hyp, internal in pairs
        ]


@dataclass
class _InvestigationFindingInternal:
    """Internal finding from a single hypothesis investigation."""
    status: str
    updated_confidence: float
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    ruling_out_step: str = ""
