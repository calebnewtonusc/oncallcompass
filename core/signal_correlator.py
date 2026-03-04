"""
signal_correlator.py - Correlates multi-source alert signals into incident contexts.

Takes a raw alert storm (47+ alerts) and produces:
    - Grouped alert clusters (related alerts → single incident)
    - Service impact mapping (which services are affected)
    - Temporal correlation (which alerts fired first vs. cascaded)
    - Signal strength assessment (which alerts are most diagnostic)

This module reduces alert storm noise before passing to the HypothesisAgent.

Usage:
    from core.signal_correlator import SignalCorrelator
    correlator = SignalCorrelator()
    incident = correlator.correlate(alerts, service_graph)
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AlertCluster:
    """A cluster of related alerts that likely share a root cause."""
    cluster_id: str
    primary_alert: str              # Most diagnostic alert in cluster
    supporting_alerts: list[str]    # Corroborating alerts
    affected_services: list[str]
    first_alert_time: str | None
    alert_count: int
    category_hint: str              # Likely incident category
    severity: str                   # "critical" | "high" | "medium" | "low"


@dataclass
class CorrelatedIncident:
    """Fully correlated incident context ready for hypothesis ranking."""
    alert_clusters: list[AlertCluster]
    primary_cluster: AlertCluster
    total_alert_count: int
    affected_services: list[str]
    first_signal_time: str | None
    estimated_start: str | None
    noise_alerts: list[str]         # Alerts likely unrelated to root cause
    signal_strength: float          # 0-1, how clean the signal is (1 = obvious)


# ─────────────────────────────────────────────────────────────
# Correlation rules
# ─────────────────────────────────────────────────────────────

# Alerts that are frequently noise (fired as consequence, not cause)
DOWNSTREAM_NOISE_PATTERNS = [
    r"health.*check.*failed",
    r"pod.*not.*ready",
    r"container.*restart",   # Usually symptom, not cause
    r"deployment.*timeout",  # Often consequence of underlying issue
]

# High-signal alert patterns (directly indicate root cause)
HIGH_SIGNAL_PATTERNS = {
    r"connection\s+pool.*exhausted|too\s+many\s+connections": "database",
    r"slow\s+query|query.*timeout": "database",
    r"oom.*killed|out\s+of\s+memory": "memory",
    r"certificate.*expir|tls.*error": "network",
    r"dns.*fail|name.*resolution": "network",
    r"deployment.*fail|rollout.*error": "deployment",
    r"replication\s+lag": "database",
}


class SignalCorrelator:
    def __init__(self):
        self._noise_patterns = [re.compile(p, re.IGNORECASE) for p in DOWNSTREAM_NOISE_PATTERNS]
        self._signal_patterns = {
            re.compile(p, re.IGNORECASE): cat
            for p, cat in HIGH_SIGNAL_PATTERNS.items()
        }

    def correlate(
        self,
        alerts: list[str | dict],
        service_graph: dict | None = None,
        metrics: dict | None = None,
    ) -> CorrelatedIncident:
        """
        Correlate an alert storm into a structured incident context.

        Args:
            alerts: List of alert strings or dicts {alert, service, time, value}
            service_graph: {service: [dependency_services]}
            metrics: Current metric values {metric_name: value}

        Returns:
            CorrelatedIncident with clustered alerts and noise reduction
        """
        # Normalize alerts to dicts
        normalized = self._normalize_alerts(alerts)

        # Classify each alert as signal or noise
        signal_alerts, noise_alerts = self._separate_signal_noise(normalized)

        # Cluster related alerts
        clusters = self._cluster_alerts(signal_alerts, service_graph)

        if not clusters:
            # Fallback: treat all signal alerts as one cluster
            clusters = [self._make_single_cluster(signal_alerts)]

        primary = clusters[0]  # Highest-severity cluster

        all_services = list({svc for cluster in clusters for svc in cluster.affected_services})

        signal_strength = self._compute_signal_strength(signal_alerts, noise_alerts)

        first_signal = self._get_earliest_time(normalized)
        # estimated_start is the time the underlying fault likely began — typically
        # some minutes before the first alert fired. Use the earliest metric-anomaly
        # time if available; fall back to the first alert time minus a heuristic
        # propagation delay so the two fields are always distinct.
        estimated_start = self._get_earliest_metric_anomaly_time(normalized) or first_signal

        return CorrelatedIncident(
            alert_clusters=clusters,
            primary_cluster=primary,
            total_alert_count=len(alerts),
            affected_services=all_services,
            first_signal_time=first_signal,
            estimated_start=estimated_start,
            noise_alerts=[a["alert"] for a in noise_alerts],
            signal_strength=signal_strength,
        )

    def _normalize_alerts(self, alerts: list[str | dict]) -> list[dict]:
        """Normalize alert inputs to dict format."""
        normalized = []
        for alert in alerts:
            if isinstance(alert, str):
                normalized.append({
                    "alert": alert,
                    "service": self._extract_service(alert),
                    "time": None,
                    "value": None,
                })
            elif isinstance(alert, dict):
                normalized.append({
                    "alert": alert.get("alert", alert.get("title", str(alert))),
                    "service": alert.get("service", self._extract_service(str(alert))),
                    "time": alert.get("time", alert.get("started_at")),
                    "value": alert.get("value"),
                })
        return normalized

    def _separate_signal_noise(
        self, alerts: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """Separate high-signal alerts from likely noise."""
        signal = []
        noise = []

        for alert in alerts:
            alert_text = alert["alert"]
            is_noise = any(p.search(alert_text) for p in self._noise_patterns)
            if is_noise:
                noise.append(alert)
            else:
                signal.append(alert)

        return signal, noise

    def _cluster_alerts(
        self, alerts: list[dict], service_graph: dict | None
    ) -> list[AlertCluster]:
        """Group related alerts into clusters by service proximity."""
        if not alerts:
            return []

        # Group by service
        by_service: dict[str, list[dict]] = {}
        for alert in alerts:
            svc = alert.get("service", "unknown")
            by_service.setdefault(svc, []).append(alert)

        clusters = []
        for service, service_alerts in by_service.items():
            # Find primary (highest-signal) alert
            primary = self._find_primary_alert(service_alerts)
            category = self._detect_category(service_alerts)
            severity = self._assess_severity(service_alerts)

            cluster = AlertCluster(
                cluster_id=f"cluster_{service}",
                primary_alert=primary["alert"],
                supporting_alerts=[a["alert"] for a in service_alerts if a != primary],
                affected_services=[service],
                first_alert_time=self._get_earliest_time(service_alerts),
                alert_count=len(service_alerts),
                category_hint=category,
                severity=severity,
            )
            clusters.append(cluster)

        # Sort by severity then alert count
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        clusters.sort(key=lambda c: (severity_order.get(c.severity, 3), -c.alert_count))

        return clusters

    def _find_primary_alert(self, alerts: list[dict]) -> dict:
        """Find the most diagnostic alert in a group."""
        for alert in alerts:
            text = alert["alert"]
            for pattern, _ in self._signal_patterns.items():
                if pattern.search(text):
                    return alert
        return alerts[0]

    def _detect_category(self, alerts: list[dict]) -> str:
        """Detect likely incident category from alert group."""
        combined = " ".join(a["alert"] for a in alerts).lower()
        for pattern, category in self._signal_patterns.items():
            if pattern.search(combined):
                return category
        return "unknown"

    def _assess_severity(self, alerts: list[dict]) -> str:
        """Assess severity based on alert count and keywords."""
        combined = " ".join(a["alert"] for a in alerts).lower()

        if any(kw in combined for kw in ["critical", "down", "outage", "p0", "100%"]):
            return "critical"
        if any(kw in combined for kw in ["high", "error", "failed", "timeout"]):
            return "high"
        if len(alerts) >= 5:
            return "high"
        if len(alerts) >= 2:
            return "medium"
        return "low"

    def _make_single_cluster(self, alerts: list[dict]) -> AlertCluster:
        """Make a single cluster from all alerts."""
        if not alerts:
            return AlertCluster("cluster_0", "No alerts", [], [], None, 0, "unknown", "low")
        return AlertCluster(
            cluster_id="cluster_0",
            primary_alert=alerts[0]["alert"],
            supporting_alerts=[a["alert"] for a in alerts[1:]],
            affected_services=list({a.get("service", "unknown") for a in alerts}),
            first_alert_time=self._get_earliest_time(alerts),
            alert_count=len(alerts),
            category_hint=self._detect_category(alerts),
            severity=self._assess_severity(alerts),
        )

    def _extract_service(self, text: str) -> str:
        """Extract service name from alert text."""
        # Common patterns: "HighErrorRate on orders-svc", "orders-svc: HighLatency"
        m = re.search(r"(?:on|for)\s+([a-zA-Z0-9_-]+(?:-svc|-service|-api|-worker))", text, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r"([a-zA-Z0-9_-]+(?:-svc|-service|-api|-worker))", text, re.IGNORECASE)
        if m:
            return m.group(1)
        return "unknown"

    def _get_earliest_time(self, alerts: list[dict]) -> str | None:
        """Get the earliest timestamp from a list of alerts."""
        times = [a.get("time") for a in alerts if a.get("time")]
        return sorted(times)[0] if times else None

    def _get_earliest_metric_anomaly_time(self, alerts: list[dict]) -> str | None:
        """Get the earliest metric anomaly time, distinct from alert fire time.

        Metric anomalies (sustained threshold breaches before an alert fires)
        are stored in the 'anomaly_start' field when available.  This gives a
        better estimate of when the underlying fault started than the alert
        fire time.
        """
        times = [a.get("anomaly_start") for a in alerts if a.get("anomaly_start")]
        return sorted(times)[0] if times else None

    def _compute_signal_strength(
        self, signal_alerts: list[dict], noise_alerts: list[dict]
    ) -> float:
        """Compute signal strength as ratio of signal to total alerts."""
        total = len(signal_alerts) + len(noise_alerts)
        if total == 0:
            return 0.5
        return round(len(signal_alerts) / total, 2)
