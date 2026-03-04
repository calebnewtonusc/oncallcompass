"""
incident_taxonomy.py - Incident category taxonomy and failure signature library.

Provides:
    - IncidentCategory enum: the 6 primary incident categories
    - FAILURE_SIGNATURES: alert patterns → likely category mapping
    - CAUSAL_GRAPH: typical causation chains for each category
    - SeverityLevel: P0-P3 severity definitions

Used by:
    - HypothesisAgent: to classify and pre-rank hypotheses
    - InvestigationAgent: to select relevant runbooks
    - CompassBench: to categorize drill scenarios
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any


class IncidentCategory(Enum):
    DATABASE = "database"
    NETWORK = "network"
    MEMORY = "memory"
    DEPLOYMENT = "deployment"
    CASCADE = "cascade"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    P0 = "P0"  # Complete outage, all users affected, exec notification
    P1 = "P1"  # Partial outage, >25% users affected, on-call + manager
    P2 = "P2"  # Degraded performance, <25% users affected, on-call
    P3 = "P3"  # Minor issue, no user impact, business hours only


@dataclass
class FailureSignature:
    """Signature pattern that identifies an incident category."""

    category: IncidentCategory
    alert_keywords: list[str]  # Keywords in alert titles
    metric_patterns: dict[str, Any]  # {metric_name: condition}
    log_patterns: list[str]  # Regex patterns in logs
    confidence_boost: float  # How strongly this signature indicates the category
    description: str


# ─────────────────────────────────────────────────────────────
# Failure Signatures — maps observable signals to categories
# ─────────────────────────────────────────────────────────────

FAILURE_SIGNATURES: list[FailureSignature] = [
    # Database signatures
    FailureSignature(
        category=IncidentCategory.DATABASE,
        alert_keywords=[
            "slow query",
            "connection pool",
            "replication lag",
            "deadlock",
            "postgres",
            "mysql",
            "database",
            "db_",
            "query",
        ],
        metric_patterns={"db_latency_ms": ">500", "db_connections": "near_max"},
        log_patterns=[
            r"connection pool.*exhausted",
            r"deadlock.*detected",
            r"could not connect.*database",
            r"too many clients",
            r"ERROR.*relation.*does not exist",
        ],
        confidence_boost=0.25,
        description="Database-related failure (slow queries, connections, replication)",
    ),
    # Network signatures
    FailureSignature(
        category=IncidentCategory.NETWORK,
        alert_keywords=[
            "dns",
            "tls",
            "ssl",
            "certificate",
            "network",
            "timeout",
            "connection refused",
            "unreachable",
            "bgp",
        ],
        metric_patterns={"tcp_errors": ">0", "dns_resolution_ms": ">200"},
        log_patterns=[
            r"connection\s+refused",
            r"name\s+resolution\s+failed",
            r"certificate\s+(expired|invalid)",
            r"network.*unreachable",
            r"ETIMEDOUT",
        ],
        confidence_boost=0.20,
        description="Network-related failure (DNS, TLS, timeouts, routing)",
    ),
    # Memory signatures
    FailureSignature(
        category=IncidentCategory.MEMORY,
        alert_keywords=[
            "oom",
            "memory",
            "heap",
            "oomkilled",
            "out of memory",
            "gc pressure",
            "swap",
        ],
        metric_patterns={"memory_pct": ">90", "gc_pause_ms": ">500"},
        log_patterns=[
            r"out\s+of\s+memory",
            r"OOMKilled",
            r"GC\s+overhead\s+limit\s+exceeded",
            r"java\.lang\.OutOfMemoryError",
            r"Cannot\s+allocate\s+memory",
        ],
        confidence_boost=0.30,
        description="Memory pressure or OOM (leak, unbounded data, GC)",
    ),
    # Deployment signatures
    FailureSignature(
        category=IncidentCategory.DEPLOYMENT,
        alert_keywords=[
            "deploy",
            "rollout",
            "release",
            "version",
            "update",
            "migration",
            "config change",
        ],
        metric_patterns={},
        log_patterns=[
            r"deployment.*started",
            r"rolling\s+update",
            r"version\s+mismatch",
            r"migration\s+failed",
        ],
        confidence_boost=0.35,  # High boost: deploy correlation is very diagnostic
        description="Deployment or configuration change caused incident",
    ),
    # Cascading failure signatures
    FailureSignature(
        category=IncidentCategory.CASCADE,
        alert_keywords=[
            "cascade",
            "downstream",
            "upstream",
            "circuit breaker",
            "dependency",
            "multiple services",
        ],
        metric_patterns={},
        log_patterns=[
            r"circuit\s+breaker.*open",
            r"upstream\s+service.*failed",
            r"dependency.*unavailable",
        ],
        confidence_boost=0.15,
        description="Cascading failure across multiple services",
    ),
    # External dependency signatures
    FailureSignature(
        category=IncidentCategory.EXTERNAL,
        alert_keywords=[
            "stripe",
            "twilio",
            "sendgrid",
            "aws",
            "gcp",
            "azure",
            "third.party",
            "external",
            "cdn",
            "payment",
        ],
        metric_patterns={},
        log_patterns=[
            r"external.*api.*error",
            r"third.party.*timeout",
            r"provider.*unavailable",
        ],
        confidence_boost=0.20,
        description="External dependency failure (payment, email, cloud provider)",
    ),
]


# ─────────────────────────────────────────────────────────────
# Causal graph — typical causation chains per category
# ─────────────────────────────────────────────────────────────

CAUSAL_GRAPH: dict[str, list[list[str]]] = {
    IncidentCategory.DATABASE.value: [
        [
            "missing_index",
            "seq_scan",
            "slow_query",
            "connection_pool_exhaustion",
            "api_timeout",
        ],
        ["connection_leak", "pool_exhaustion", "request_queuing", "latency_spike"],
        [
            "replication_lag",
            "read_replica_stale",
            "cache_invalidation_storm",
            "db_overload",
        ],
    ],
    IncidentCategory.MEMORY.value: [
        [
            "memory_leak",
            "gradual_exhaustion",
            "oom_kill",
            "pod_restart",
            "traffic_spike_on_restart",
        ],
        ["large_query", "unbounded_result_set", "memory_spike", "oom_kill"],
    ],
    IncidentCategory.DEPLOYMENT.value: [
        ["bad_deploy", "code_bug", "error_rate_spike"],
        ["config_change", "misconfiguration", "service_crash"],
        ["database_migration", "schema_change", "query_failure"],
    ],
    IncidentCategory.CASCADE.value: [
        [
            "service_a_failure",
            "service_b_timeout",
            "connection_pool_b_exhaustion",
            "service_c_errors",
        ],
        ["database_slow", "api_latency", "frontend_timeout", "user_facing_errors"],
    ],
}


def classify_from_signals(
    alerts: list[str], metrics: dict, logs: str = ""
) -> IncidentCategory:
    """
    Classify incident category from observable signals.

    Uses FAILURE_SIGNATURES to score each category and return the most likely.
    """
    scores: dict[IncidentCategory, float] = {cat: 0.0 for cat in IncidentCategory}

    combined_text = " ".join(alerts).lower() + " " + logs.lower()

    for sig in FAILURE_SIGNATURES:
        # Check alert keywords
        keyword_hits = sum(1 for kw in sig.alert_keywords if kw in combined_text)
        if keyword_hits > 0:
            scores[sig.category] += sig.confidence_boost * (
                keyword_hits / len(sig.alert_keywords)
            )

        # Check metric patterns (simple threshold check)
        for metric, condition in sig.metric_patterns.items():
            if metric in metrics:
                scores[sig.category] += sig.confidence_boost * 0.5

    best = max(scores.items(), key=lambda x: x[1])
    if best[1] < 0.05:
        return IncidentCategory.UNKNOWN

    return best[0]


def get_causal_chains(category: IncidentCategory) -> list[list[str]]:
    """Get typical causation chains for an incident category."""
    return CAUSAL_GRAPH.get(category.value, [])
