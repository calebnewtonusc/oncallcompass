"""
runbook_synthesizer.py - Synthesizes runbook entries for common failure patterns.

Generates structured runbook content that teaches OncallCompass the
investigation sequence for each alert type, not just the root cause.

The key insight: runbooks encode SRE institutional knowledge about
investigation ORDER — which checks are highest signal for each failure mode.

Usage:
    from synthesis.runbook_synthesizer import RunbookSynthesizer
    synth = RunbookSynthesizer()
    runbooks = synth.generate_all()
"""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from loguru import logger

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class RunbookEntry:
    """A complete runbook entry with investigation sequence."""

    alert_type: str
    service_category: str
    symptoms: list[str]
    first_check: str  # The single highest-signal first check
    investigation_sequence: list[dict]  # [{step, rationale, expected_finding}]
    decision_tree: dict  # Conditional investigation path
    escalation_criteria: list[str]
    common_causes: list[str]
    false_positive_patterns: list[str]
    estimated_mttr_min: int  # Expected MTTR in minutes


# ─────────────────────────────────────────────────────────────
# Pre-encoded expert runbooks (no LLM needed)
# ─────────────────────────────────────────────────────────────

EXPERT_RUNBOOKS: list[RunbookEntry] = [
    RunbookEntry(
        alert_type="HighErrorRate",
        service_category="web_api",
        symptoms=[
            "HTTP 5xx rate > 1% for 5 minutes",
            "Error rate spike on specific endpoint(s)",
        ],
        first_check="Deployment timeline: was there a deploy in the last 2 hours?",
        investigation_sequence=[
            {
                "step": "Check recent deployments: `git log --oneline -10` or CI/CD dashboard",
                "rationale": "80% of error rate spikes correlate with recent deploys",
                "expected_finding": "Deployment 30-90 minutes ago = deployment regression",
            },
            {
                "step": "Break down 5xx by error type: 500 (app bug) vs 502 (upstream gateway) vs 503 (pool/circuit breaker) vs 504 (timeout)",
                "rationale": "Error type determines investigation path",
                "expected_finding": "503 → connection pool / circuit breaker; 504 → upstream latency",
            },
            {
                "step": "Check upstream service health: database, cache, external APIs",
                "rationale": "Upstream degradation causes downstream 5xx",
                "expected_finding": "Database latency spike → connection pool investigation",
            },
            {
                "step": "Check application error logs for stack trace",
                "rationale": "Stack trace identifies the specific failing code path",
                "expected_finding": "NullPointerException or connection timeout",
            },
        ],
        decision_tree={
            "recent_deploy": "Roll back and verify recovery",
            "upstream_degraded": "Escalate to infrastructure team",
            "no_recent_deploy_no_upstream": "Check for data anomaly or traffic spike",
        },
        escalation_criteria=[
            "Error rate > 10% for > 15 minutes",
            "Affects payment or checkout flows",
            "No root cause found after 20 minutes",
        ],
        common_causes=[
            "Deployment regression (code bug in new version)",
            "Upstream dependency timeout",
            "Database connection pool exhaustion",
            "OOM causing pod restart loop",
        ],
        false_positive_patterns=[
            "Single client with broken retry logic",
            "Load balancer health check misconfiguration",
            "Monitoring agent misconfiguration",
        ],
        estimated_mttr_min=25,
    ),
    RunbookEntry(
        alert_type="HighLatency",
        service_category="web_api",
        symptoms=[
            "P99 latency > 2x baseline for 5 minutes",
            "P50 latency normal, P99 elevated (tail latency issue)",
        ],
        first_check="Is CPU elevated alongside latency? Yes → compute bound. No → I/O or lock contention.",
        investigation_sequence=[
            {
                "step": "Check CPU vs latency correlation: if CPU < 70% but latency high → I/O bound",
                "rationale": "Separates compute-bound from I/O-bound root causes immediately",
                "expected_finding": "Low CPU + high latency = database, cache, or network I/O",
            },
            {
                "step": "Check database slow query log for queries > 1s",
                "rationale": "Slow queries are the most common cause of API latency spikes",
                "expected_finding": "Missing index on recently modified table = seq scan on large table",
            },
            {
                "step": "Check cache hit ratio: drop in cache hit rate = cold cache or thundering herd",
                "rationale": "Cache miss storm forces all requests to hit database",
                "expected_finding": "Hit ratio drop from 95% to 60% = cache invalidation or restart",
            },
            {
                "step": "Check thread/connection pool metrics: waiting connection count",
                "rationale": "Pool exhaustion causes latency to spike as requests queue",
                "expected_finding": "Pool max_connections reached = all new requests waiting",
            },
        ],
        decision_tree={
            "high_cpu": "Profile code — likely hot loop or regex",
            "slow_db_queries": "Add missing index or kill long-running queries",
            "low_cache_hit": "Investigate cache restart or TTL misconfiguration",
            "pool_exhaustion": "Increase pool size and find connection leak",
        },
        escalation_criteria=[
            "P99 > 10x baseline",
            "User-facing timeout errors appearing",
            "No improvement after standard mitigation",
        ],
        common_causes=[
            "Missing index on frequently queried table",
            "Cache stampede after restart",
            "Connection pool exhaustion",
            "GC pressure in JVM service",
            "N+1 query problem in new code",
        ],
        false_positive_patterns=[
            "Synthetic monitoring from different geographic region",
            "Batch job running during business hours",
        ],
        estimated_mttr_min=35,
    ),
    RunbookEntry(
        alert_type="OOMKilled",
        service_category="container",
        symptoms=[
            "Pod OOMKilled restart",
            "Container memory usage hit limit",
        ],
        first_check="Is this a gradual memory increase (leak) or sudden spike (large data load)?",
        investigation_sequence=[
            {
                "step": "Check memory usage graph before OOM: gradual increase or sudden spike?",
                "rationale": "Pattern determines root cause type",
                "expected_finding": "Gradual → memory leak; sudden → large data loading",
            },
            {
                "step": "Check for recent deployment: new code often introduces memory regressions",
                "rationale": "Memory leaks frequently introduced in deployments",
                "expected_finding": "OOM timing correlated with deploy time = deployment regression",
            },
            {
                "step": "Check for unbounded queries: `SELECT *` without LIMIT, large file uploads",
                "rationale": "Loading large datasets into memory causes OOM",
                "expected_finding": "New API endpoint loading full table into memory",
            },
            {
                "step": "For JVM: check heap dump or GC logs for object accumulation",
                "rationale": "JVM heap dump shows which objects are accumulating",
                "expected_finding": "HashMap growing unbounded = reference not being cleared",
            },
        ],
        decision_tree={
            "gradual_leak": "Deploy fix or roll back; increase memory limit as temporary",
            "sudden_spike": "Find and fix unbounded data loading; add LIMIT to queries",
            "jvm_leak": "Take heap dump, analyze with MAT or VisualVM",
        },
        escalation_criteria=[
            "Pod restart loop (CrashLoopBackOff)",
            "Multiple pods OOM simultaneously",
            "No root cause after 30 minutes",
        ],
        common_causes=[
            "Memory leak in new deployment",
            "Missing LIMIT on database query",
            "Large file upload loaded entirely into memory",
            "JVM heap not tuned for container memory limit",
            "Connection objects not closed",
        ],
        false_positive_patterns=[
            "Scheduled batch job with expected high memory use",
            "Memory limit set too low for normal operation",
        ],
        estimated_mttr_min=40,
    ),
]


class RunbookSynthesizer:
    def __init__(self):
        self.client = None
        if HAS_ANTHROPIC:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)

    def generate_all(self) -> list[RunbookEntry]:
        """Return all pre-encoded expert runbooks + any LLM-synthesized extras."""
        runbooks = list(EXPERT_RUNBOOKS)

        if self.client:
            extra = self._synthesize_additional()
            runbooks.extend(extra)

        return runbooks

    def _synthesize_additional(self) -> list[RunbookEntry]:
        """Use Claude to synthesize additional runbook entries."""
        additional_alert_types = [
            ("DiskFull", "database", "PostgreSQL transaction log filling disk"),
            ("HighQueueDepth", "message_queue", "Kafka consumer lag growing"),
            ("CertificateExpiry", "networking", "TLS certificate expiring in <48h"),
            ("ReplicationLag", "database", "PostgreSQL replica falling behind primary"),
        ]

        entries = []
        for alert_type, category, scenario in additional_alert_types:
            try:
                entry = self._synthesize_single(alert_type, category, scenario)
                if entry:
                    entries.append(entry)
            except Exception as e:
                logger.debug(f"Failed to synthesize runbook for {alert_type}: {e}")

        return entries

    def _synthesize_single(
        self, alert_type: str, category: str, scenario: str
    ) -> RunbookEntry | None:
        if not self.client:
            return None

        prompt = f"""Generate a structured SRE runbook for this alert type:

Alert Type: {alert_type}
Service Category: {category}
Scenario: {scenario}

Return a JSON object with these fields:
- alert_type, service_category, symptoms (list), first_check (string)
- investigation_sequence: list of {{step, rationale, expected_finding}}
- decision_tree: dict of condition → action
- escalation_criteria (list), common_causes (list), false_positive_patterns (list)
- estimated_mttr_min (int)

Focus on investigation ORDER — what a senior SRE checks FIRST and WHY.
Return ONLY valid JSON."""

        try:
            resp = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system="You are a senior SRE creating runbook documentation.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            import re

            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group())
            # Use type-correct empty defaults rather than borrowing values from
            # an expert runbook, which would silently inject stale runbook data
            # into a synthetically generated entry when fields are missing.
            return RunbookEntry(
                alert_type=data.get("alert_type", alert_type),
                service_category=data.get("service_category", category),
                symptoms=data.get("symptoms", []),
                first_check=data.get("first_check", ""),
                investigation_sequence=data.get("investigation_sequence", []),
                decision_tree=data.get("decision_tree", {}),
                escalation_criteria=data.get("escalation_criteria", []),
                common_causes=data.get("common_causes", []),
                false_positive_patterns=data.get("false_positive_patterns", []),
                estimated_mttr_min=int(data.get("estimated_mttr_min", 30)),
            )
        except Exception as e:
            logger.debug(f"Runbook synthesis error: {e}")
            return None

    def to_training_pairs(self, runbooks: list[RunbookEntry]) -> list[dict]:
        """Convert runbooks to training pair format."""
        pairs = []
        for rb in runbooks:
            # Create an incident scenario from the runbook
            pair = {
                "alerts": rb.symptoms,
                "context": {
                    "alert_type": rb.alert_type,
                    "service_category": rb.service_category,
                },
                "ranked_hypotheses": [
                    {
                        "hypothesis": cause,
                        "confidence": 0.7 - (i * 0.1),
                        "evidence": [],
                        "ruling_out": rb.investigation_sequence[0]["step"]
                        if rb.investigation_sequence
                        else "",
                    }
                    for i, cause in enumerate(rb.common_causes[:3])
                ],
                "investigation_steps": [s["step"] for s in rb.investigation_sequence],
                "postmortem_draft": {
                    "summary": f"{rb.alert_type} on {rb.service_category}",
                    "timeline": [],
                    "root_cause": rb.common_causes[0] if rb.common_causes else "",
                    "contributing_factors": rb.false_positive_patterns[:2],
                    "action_items": [
                        {
                            "item": f"Add monitoring for {rb.alert_type}",
                            "owner": "SRE",
                            "prevents": "Late detection",
                        }
                    ],
                },
            }
            pairs.append(pair)
        return pairs

    def save(
        self, runbooks: list[RunbookEntry], output_path: str = "data/raw/runbooks.jsonl"
    ) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for rb in runbooks:
                f.write(json.dumps(asdict(rb)) + "\n")
        logger.info(f"Saved {len(runbooks)} runbooks to {output_path}")


if __name__ == "__main__":
    synth = RunbookSynthesizer()
    runbooks = synth.generate_all()
    synth.save(runbooks)
    print(f"Generated {len(runbooks)} runbook entries")
