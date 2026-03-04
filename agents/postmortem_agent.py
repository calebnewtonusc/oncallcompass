"""
postmortem_agent.py - Generates structured 5-Why postmortems from confirmed root causes.

The Postmortem Agent:
1. Receives the confirmed root cause from the Investigation Agent
2. Constructs the incident timeline from alert timestamps
3. Generates a complete 5-Why analysis
4. Produces actionable remediation items with owners

5-Why format:
    Why 1: Why did the service fail?
    Why 2: Why did that happen?
    Why 3: Why did that happen?
    Why 4: Why was this not prevented?
    Why 5: Why was there no detection/alerting before impact?

Usage:
    from agents.postmortem_agent import PostmortemAgent
    agent = PostmortemAgent()
    pm = agent.generate(incident, confirmed_root_cause)
    print(pm.five_why_analysis)
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anthropic
from loguru import logger


def _parse_int_field(value: object) -> int:
    """Parse an integer field that may arrive as an int or a string like '45 minutes'."""
    if isinstance(value, int):
        return value
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else 0


@dataclass
class FiveWhyAnalysis:
    """Structured 5-Why root cause analysis."""
    why_1: str  # Service failure symptom
    why_2: str  # Immediate cause
    why_3: str  # System cause
    why_4: str  # Process/prevention gap
    why_5: str  # Detection/alerting gap
    root_cause: str  # The final "why" — systemic root cause


@dataclass
class ActionItem:
    """A single postmortem action item."""
    item: str
    owner: str        # Team or system layer responsible
    prevents: str     # What recurrence this prevents
    priority: str     # "immediate" | "short_term" | "long_term"
    deadline: str     # e.g., "within 1 week"


@dataclass
class Postmortem:
    """Complete structured postmortem."""
    title: str
    severity: str        # P0/P1/P2/P3
    start_time: str
    end_time: str
    duration_minutes: int
    affected_services: list[str]
    user_impact: str

    executive_summary: str
    timeline: list[dict]   # [{time, event, type}]
    five_why: FiveWhyAnalysis
    root_cause: str
    contributing_factors: list[str]
    action_items: list[ActionItem]

    detection_time_min: int   # Time from incident start to alert
    ttr_min: int              # Time to resolve
    mttr_baseline_min: int    # Baseline MTTR for this incident type


POSTMORTEM_SYSTEM = """You are OncallCompass's Postmortem Writer — an expert at turning incident data
into structured, prevention-focused postmortems.

A good postmortem:
1. Has a 5-Why analysis that reaches SYSTEMIC root cause (not "the code had a bug")
2. Has action items that PREVENT recurrence, not just fix symptoms
3. Has a timeline that tells the story of the incident chronologically
4. Has user impact that is specific (not "some users affected")

5-Why rules:
- Why 1: Symptom (what the user/monitor saw)
- Why 2: Immediate technical cause
- Why 3: Why the immediate cause happened
- Why 4: Why the system/process allowed this (gap in testing, monitoring, runbook)
- Why 5: What prevented earlier detection or faster response

Action items must have:
- specific owner (not "engineering team" — be specific: "SRE team", "Payments backend team")
- prevents: what recurrence scenario this action item eliminates
- priority: immediate (this week), short_term (this quarter), long_term (roadmap)

Return ONLY valid JSON."""


class PostmortemAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    def generate(
        self,
        incident: dict,
        confirmed_root_cause: str,
        investigation_steps: list[str] | None = None,
    ) -> Postmortem:
        """
        Generate a complete postmortem from confirmed incident data.

        Args:
            incident: Full incident context (alerts, logs, metrics, context, timeline)
            confirmed_root_cause: The confirmed root cause string
            investigation_steps: Steps taken during investigation

        Returns:
            Complete Postmortem with 5-Why analysis and action items
        """
        prompt = self._build_prompt(incident, confirmed_root_cause, investigation_steps)

        logger.info(f"Generating postmortem for: {confirmed_root_cause[:80]}")

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            system=POSTMORTEM_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text

        return self._parse_postmortem(text, incident, confirmed_root_cause)

    def _build_prompt(
        self,
        incident: dict,
        confirmed_root_cause: str,
        investigation_steps: list[str] | None,
    ) -> str:
        import json
        parts = []

        alerts = incident.get("alerts", [])
        ctx = incident.get("context", {})

        parts.append(f"## Confirmed Root Cause\n{confirmed_root_cause}")
        parts.append(f"## Alert Signals\n" + "\n".join(f"- {a}" for a in alerts))

        if ctx.get("stack"):
            parts.append(f"## Affected Stack\n{', '.join(ctx['stack'])}")

        if ctx.get("last_deploy"):
            parts.append(f"## Last Deploy\n{ctx['last_deploy']}")

        if incident.get("metrics"):
            parts.append(f"## Peak Metrics\n{json.dumps(incident['metrics'], indent=2)}")

        if incident.get("logs"):
            parts.append(f"## Representative Logs\n```\n{incident['logs'][:600]}\n```")

        if investigation_steps:
            parts.append("## Investigation Steps Taken\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(investigation_steps)))

        if incident.get("timeline"):
            parts.append("## Incident Timeline\n" + "\n".join(f"- {t}" for t in incident["timeline"][:10]))

        parts.append(
            "\nGenerate a complete structured postmortem in JSON format. "
            "The 5-Why analysis must reach SYSTEMIC root cause. "
            "Action items must PREVENT recurrence, not just fix symptoms."
        )

        return "\n\n".join(parts)

    def _parse_postmortem(
        self, text: str, incident: dict, root_cause: str
    ) -> Postmortem:
        """Parse LLM response into Postmortem dataclass."""
        import json

        # Try to parse structured JSON — use greedy match with DOTALL so we
        # capture the full nested JSON object rather than truncating at the
        # first closing brace.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                return self._build_from_json(data, incident, root_cause)
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: build from incident context
        return self._build_fallback(incident, root_cause, text)

    def _build_from_json(self, data: dict, incident: dict, root_cause: str) -> Postmortem:
        """Build Postmortem from parsed JSON response."""
        five_why_data = data.get("five_why", {})
        five_why = FiveWhyAnalysis(
            why_1=five_why_data.get("why_1", "Service failure detected"),
            why_2=five_why_data.get("why_2", ""),
            why_3=five_why_data.get("why_3", ""),
            why_4=five_why_data.get("why_4", ""),
            why_5=five_why_data.get("why_5", ""),
            root_cause=data.get("root_cause", root_cause),
        )

        action_items = [
            ActionItem(
                item=ai.get("item", ""),
                owner=ai.get("owner", "Engineering"),
                prevents=ai.get("prevents", ""),
                priority=ai.get("priority", "short_term"),
                deadline=ai.get("deadline", "within 2 weeks"),
            )
            for ai in data.get("action_items", [])
        ]

        return Postmortem(
            title=data.get("title", f"Incident: {root_cause[:60]}"),
            severity=data.get("severity", "P2"),
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time", ""),
            duration_minutes=_parse_int_field(data.get("duration_minutes", 0)),
            affected_services=data.get("affected_services", incident.get("context", {}).get("stack", [])),
            user_impact=data.get("user_impact", ""),
            executive_summary=data.get("executive_summary", data.get("summary", "")),
            timeline=data.get("timeline", []),
            five_why=five_why,
            root_cause=root_cause,
            contributing_factors=data.get("contributing_factors", []),
            action_items=action_items,
            detection_time_min=_parse_int_field(data.get("detection_time_min", 5)),
            ttr_min=_parse_int_field(data.get("ttr_min", 30)),
            mttr_baseline_min=30,
        )

    def _build_fallback(self, incident: dict, root_cause: str, llm_text: str) -> Postmortem:
        """Build a minimal postmortem when JSON parsing fails."""
        ctx = incident.get("context", {})
        stack = ctx.get("stack", [])

        # Extract action items from text
        action_items = []
        for m in re.finditer(r"[-•]\s+(.{20,200})", llm_text):
            item_text = m.group(1).strip()
            if any(kw in item_text.lower() for kw in ["add", "implement", "create", "fix", "update", "monitor"]):
                action_items.append(ActionItem(
                    item=item_text[:150],
                    owner="Engineering",
                    prevents="Recurrence of this incident type",
                    priority="short_term",
                    deadline="within 2 weeks",
                ))

        if not action_items:
            action_items = [ActionItem(
                item=f"Investigate and fix: {root_cause}",
                owner="Engineering",
                prevents="Recurrence",
                priority="immediate",
                deadline="within 1 week",
            )]

        return Postmortem(
            title=f"Incident: {root_cause[:60]}",
            severity="P2",
            start_time="",
            end_time="",
            duration_minutes=0,
            affected_services=stack,
            user_impact="See incident alerts for impact details",
            executive_summary=f"Incident caused by: {root_cause}",
            timeline=[],
            five_why=FiveWhyAnalysis(
                why_1="Service degradation detected by monitoring",
                why_2=root_cause,
                why_3="See investigation notes",
                why_4="Prevention gap identified",
                why_5="Detection gap identified",
                root_cause=root_cause,
            ),
            root_cause=root_cause,
            contributing_factors=[],
            action_items=action_items,
            detection_time_min=5,
            ttr_min=30,
            mttr_baseline_min=30,
        )
