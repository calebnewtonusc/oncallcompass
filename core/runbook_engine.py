"""
runbook_engine.py - Runbook lookup and matching engine.

Provides the InvestigationAgent with relevant investigation steps
based on incident category and service type.

Design: runbooks are stored as structured dicts with investigation
sequences. The engine matches the best runbook for a given incident
and returns the top investigation steps.

Usage:
    from core.runbook_engine import RunbookEngine
    engine = RunbookEngine()
    steps = engine.get_steps("database", "web_api")
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Runbook:
    """A single runbook entry."""
    category: str
    service_type: str
    investigation_steps: list[str]
    first_check: str
    common_causes: list[str]
    escalation_criteria: list[str]


# ─────────────────────────────────────────────────────────────
# Built-in runbook library
# ─────────────────────────────────────────────────────────────

BUILTIN_RUNBOOKS: list[Runbook] = [
    Runbook(
        category="database",
        service_type="general",
        investigation_steps=[
            "Check pg_stat_activity for long-running queries: SELECT * FROM pg_stat_activity WHERE state != 'idle' ORDER BY query_start",
            "Check connection pool metrics: max_connections vs active_connections",
            "Check database slow query log for queries > 1 second",
            "Check for table bloat: SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) FROM pg_tables ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC LIMIT 5",
            "Check replication lag: SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag",
            "Check lock contention: SELECT * FROM pg_locks l JOIN pg_stat_activity a ON l.pid = a.pid WHERE granted = false",
        ],
        first_check="Check pg_stat_activity for long-running queries and connection count",
        common_causes=[
            "Missing index causing sequential scan on large table",
            "Connection pool exhaustion from connection leak",
            "Long-running transaction holding locks",
            "Stale statistics causing bad query plan",
        ],
        escalation_criteria=[
            "Database CPU > 90% for > 10 minutes",
            "Connection pool at max capacity for > 5 minutes",
            "Replication lag > 60 seconds",
        ],
    ),
    Runbook(
        category="memory",
        service_type="container",
        investigation_steps=[
            "Check memory usage graph: gradual increase (leak) or sudden spike (large data)?",
            "Check recent deployments — was there a deploy in the last 2 hours?",
            "Check for unbounded queries: SELECT * without LIMIT",
            "Check application logs for OutOfMemoryError or GC messages",
            "For JVM: check heap dump or GC logs: jcmd <pid> GC.run_finalization",
            "Check container memory limits vs actual usage: kubectl top pod",
        ],
        first_check="Memory graph pattern: gradual increase = leak, sudden spike = large data load",
        common_causes=[
            "Memory leak in new deployment",
            "Missing LIMIT on database query loading full table",
            "JVM heap not tuned for container memory limit",
            "Connection objects not being closed properly",
        ],
        escalation_criteria=[
            "Pod in CrashLoopBackOff",
            "Multiple pods OOM simultaneously",
            "No root cause after 30 minutes",
        ],
    ),
    Runbook(
        category="deployment",
        service_type="general",
        investigation_steps=[
            "Check deployment timeline: when did the error rate spike relative to deployment?",
            "Compare error rate on new version vs old version: kubectl get pods -l version=",
            "Check for breaking API changes in deployment diff",
            "Check feature flags: were any enabled in this deployment?",
            "Initiate rollback if correlation confirmed: kubectl rollout undo deployment/<name>",
            "Post-rollback: verify error rate returns to baseline within 2 minutes",
        ],
        first_check="Was there a deployment in the last 2 hours? Correlation > 80% probability of regression.",
        common_causes=[
            "New code introduced a bug in hot path",
            "Database migration made a breaking schema change",
            "Feature flag enabled incorrect code path",
            "Container image built with wrong environment variables",
        ],
        escalation_criteria=[
            "Rollback did not resolve error rate",
            "Database migration cannot be rolled back",
            "Deployment in progress during active incident",
        ],
    ),
    Runbook(
        category="network",
        service_type="general",
        investigation_steps=[
            "Test DNS resolution: nslookup <domain> && dig <domain>",
            "Check TLS certificate expiry: openssl s_client -connect <host>:443 2>/dev/null | openssl x509 -noout -dates",
            "Check network connectivity between services: curl -v http://<service>:<port>/health",
            "Check firewall/security group changes in last 24 hours",
            "Traceroute to identify where packets are dropping: traceroute <host>",
            "Check load balancer health: verify backend pool member status",
        ],
        first_check="DNS: `nslookup <service>`. TLS: `openssl s_client -connect <host>:443`",
        common_causes=[
            "TLS certificate expired",
            "DNS misconfiguration after infrastructure change",
            "Security group rule blocking traffic",
            "Service mesh configuration error",
        ],
        escalation_criteria=[
            "Complete connectivity loss between services",
            "Certificate expired with no renewal in progress",
            "Network partition confirmed",
        ],
    ),
    Runbook(
        category="cascade",
        service_type="general",
        investigation_steps=[
            "Map the alert timeline: which service alerted FIRST? That service is likely the root cause.",
            "Trace service dependency graph: which upstream service does the failing service depend on?",
            "Check circuit breaker status: are any circuit breakers open?",
            "Check the shared dependency: database, cache, message queue",
            "Implement traffic shedding on failed service to prevent cascade spread",
            "Fix the upstream service first, then verify downstream recovery",
        ],
        first_check="Find the FIRST alert in the timeline — the root cause service usually alerts first.",
        common_causes=[
            "Shared database overloaded by one service, cascading to others",
            "Redis cluster failure causing cache miss storm across all services",
            "Slow upstream service exhausting connection pools in downstream services",
        ],
        escalation_criteria=[
            "Cascade spreading to more than 5 services",
            "Core infrastructure (database, cache) unavailable",
            "Payment or authentication service in cascade",
        ],
    ),
]

# Build lookup index
_RUNBOOK_INDEX: dict[tuple[str, str], Runbook] = {}
_CATEGORY_INDEX: dict[str, list[Runbook]] = {}

for rb in BUILTIN_RUNBOOKS:
    key = (rb.category.lower(), rb.service_type.lower())
    _RUNBOOK_INDEX[key] = rb
    _CATEGORY_INDEX.setdefault(rb.category.lower(), []).append(rb)


class RunbookEngine:
    def get_steps(self, alert_type: str, service_category: str = "general") -> list[str]:
        """
        Get investigation steps for an alert type and service category.

        Args:
            alert_type: Incident category (e.g., "database", "memory", "deployment")
            service_category: Service type (e.g., "web_api", "container", "general")

        Returns:
            List of investigation step strings
        """
        runbook = self._find_runbook(alert_type, service_category)
        if runbook:
            return runbook.investigation_steps
        return [f"Investigate {alert_type} incident on {service_category} service"]

    def get_first_check(self, alert_type: str) -> str:
        """Get the single highest-signal first check for an alert type."""
        runbook = self._find_runbook(alert_type, "general")
        return runbook.first_check if runbook else f"Check {alert_type} service logs and metrics"

    def get_runbook(self, alert_type: str, service_category: str = "general") -> Runbook | None:
        """Get the full runbook for an alert type."""
        return self._find_runbook(alert_type, service_category)

    def get_escalation_criteria(self, alert_type: str) -> list[str]:
        """Get escalation criteria for an alert type."""
        runbook = self._find_runbook(alert_type, "general")
        return runbook.escalation_criteria if runbook else []

    def _find_runbook(self, alert_type: str, service_category: str) -> Runbook | None:
        """Find the best matching runbook."""
        # Exact match
        key = (alert_type.lower(), service_category.lower())
        if key in _RUNBOOK_INDEX:
            return _RUNBOOK_INDEX[key]

        # Category match (general fallback)
        generic_key = (alert_type.lower(), "general")
        if generic_key in _RUNBOOK_INDEX:
            return _RUNBOOK_INDEX[generic_key]

        # Partial category match
        category_runbooks = _CATEGORY_INDEX.get(alert_type.lower(), [])
        if category_runbooks:
            return category_runbooks[0]

        return None

    def list_categories(self) -> list[str]:
        """List all available runbook categories."""
        return list(_CATEGORY_INDEX.keys())
