"""
pagerduty_corpus.py - Harvests incident response knowledge from PagerDuty resources.

Sources:
    - PagerDuty Incident Response Guide (open source)
    - PagerDuty Community posts
    - PagerDuty blog (incident management posts)
    - Opsgenie/Atlassian Incident Handbook

Outputs training data focused on:
    - Alert fatigue patterns (high-volume noisy alerts → single root cause)
    - Escalation path decisions
    - On-call runbook patterns
    - Retrospective analysis

Usage:
    from discovery.pagerduty_corpus import PagerDutyCorpus
    corpus = PagerDutyCorpus()
    docs = await corpus.collect()
"""

import asyncio
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger


@dataclass
class RunbookEntry:
    """A single runbook entry with alert context and resolution steps."""
    source: str
    alert_type: str          # e.g., "HighErrorRate", "OOMKilled", "DiskFull"
    service_type: str        # e.g., "web", "database", "cache", "queue"
    symptoms: list[str]
    investigation_steps: list[str]
    common_causes: list[str]
    resolution_steps: list[str]
    escalation_criteria: str


PAGERDUTY_RUNBOOK_SOURCES = [
    {
        "name": "pagerduty_irg",
        "url": "https://response.pagerduty.com/",
        "sections": [
            "during/alerting_principles",
            "during/incident_response_doc",
            "during/war_room",
            "after/post_mortem_process",
            "after/effective_post_mortems",
        ],
    },
    {
        "name": "atlassian_handbook",
        "url": "https://www.atlassian.com/incident-management/",
        "sections": [
            "postmortem",
            "on-call",
            "alerts",
            "runbook",
        ],
    },
]

# Alert type patterns commonly found in incident reports
ALERT_PATTERNS = {
    "HighErrorRate": {
        "service_types": ["web", "api", "microservice"],
        "common_causes": [
            "upstream dependency timeout",
            "database connection pool exhaustion",
            "deployment regression",
            "memory pressure causing GC pauses",
            "config change breaking validation",
        ],
        "investigation_steps": [
            "Check error rate trend — sudden vs gradual?",
            "Identify error types in logs (5xx breakdown: 500/502/503/504)",
            "Correlate with recent deployments (git log --oneline -20)",
            "Check upstream service health",
            "Check database connection pool metrics",
        ],
    },
    "HighLatency": {
        "service_types": ["web", "api", "database", "cache"],
        "common_causes": [
            "slow database queries (missing index, lock contention)",
            "cache miss storm (cold cache after restart)",
            "network congestion between services",
            "CPU throttling (container limits)",
            "GC pressure in JVM services",
        ],
        "investigation_steps": [
            "Check P50/P95/P99 percentile breakdown",
            "Identify which endpoints are slow",
            "Check database slow query log",
            "Check cache hit ratio",
            "Check CPU throttling metrics in k8s",
        ],
    },
    "OOMKilled": {
        "service_types": ["web", "worker", "batch"],
        "common_causes": [
            "memory leak in new deployment",
            "unexpected traffic spike exceeding memory limit",
            "large query result set loaded into memory",
            "misconfigured JVM heap size",
            "connection objects not being closed",
        ],
        "investigation_steps": [
            "Check which container/pod was OOM killed",
            "Compare memory usage before/after last deployment",
            "Check for heap dumps if JVM",
            "Look for unbounded data loading in recent code changes",
            "Check memory limits vs actual usage trend",
        ],
    },
    "DiskFull": {
        "service_types": ["database", "log", "worker"],
        "common_causes": [
            "log rotation misconfigured",
            "database transaction logs not being truncated",
            "large file uploads not being cleaned up",
            "WAL accumulation (PostgreSQL replication lag)",
            "core dump files accumulating",
        ],
        "investigation_steps": [
            "Run `du -sh /*` to find largest directories",
            "Check log rotation config (logrotate.d)",
            "Check database WAL/binlog size",
            "Look for large tmp files",
            "Check PostgreSQL pg_xlog or MySQL binlog retention",
        ],
    },
}


class PagerDutyCorpus:
    def __init__(self, output_dir: str = "data/raw/runbooks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def collect(self) -> list[RunbookEntry]:
        """Collect runbook entries from all sources."""
        entries = []

        # Add structured runbook entries from alert patterns
        for alert_type, pattern_data in ALERT_PATTERNS.items():
            for service_type in pattern_data["service_types"]:
                entry = RunbookEntry(
                    source="alert_pattern_library",
                    alert_type=alert_type,
                    service_type=service_type,
                    symptoms=[f"{alert_type} on {service_type} service"],
                    investigation_steps=pattern_data["investigation_steps"],
                    common_causes=pattern_data["common_causes"],
                    resolution_steps=[
                        f"Address the identified root cause for {alert_type}",
                        "Verify metrics return to normal",
                        "Document in incident log",
                    ],
                    escalation_criteria=f"Escalate if {alert_type} persists >10min or affects >5% users",
                )
                entries.append(entry)

        # Crawl online sources
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20),
            headers={"User-Agent": "OncallCompass-Research/1.0"},
        ) as session:
            web_entries = await self._crawl_web_sources(session)
            entries.extend(web_entries)

        logger.info(f"Collected {len(entries)} runbook entries")
        self._save(entries)
        return entries

    async def _crawl_web_sources(self, session: aiohttp.ClientSession) -> list[RunbookEntry]:
        """Crawl PagerDuty and Atlassian incident response guides."""
        entries = []
        for source in PAGERDUTY_RUNBOOK_SOURCES:
            for section in source.get("sections", []):
                url = f"{source['url'].rstrip('/')}/{section}"
                try:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            html = await resp.text()
                            section_entries = self._parse_runbook_page(html, source["name"], section)
                            entries.extend(section_entries)
                except Exception as e:
                    logger.debug(f"Failed to fetch {url}: {e}")

        return entries

    def _parse_runbook_page(self, html: str, source: str, section: str) -> list[RunbookEntry]:
        """Extract structured runbook content from HTML."""
        soup = BeautifulSoup(html, "html.parser")
        entries = []

        # Extract h2/h3 sections as runbook entries
        main_content = soup.find("main") or soup.find("article") or soup

        for heading in main_content.find_all(["h2", "h3"]):
            title = heading.get_text(strip=True)
            if not title or len(title) < 5:
                continue

            # Get content after this heading until next heading
            content_parts = []
            for sibling in heading.next_siblings:
                if sibling.name in ["h2", "h3"]:
                    break
                if hasattr(sibling, "get_text"):
                    content_parts.append(sibling.get_text(separator=" ", strip=True))

            content = " ".join(content_parts)
            if len(content) < 100:
                continue

            # Extract list items as steps
            steps = []
            for list_elem in heading.find_next_siblings(["ol", "ul"]):
                for li in list_elem.find_all("li"):
                    step = li.get_text(strip=True)
                    if step:
                        steps.append(step[:200])

            if steps:
                entry = RunbookEntry(
                    source=f"{source}/{section}",
                    alert_type=self._classify_alert_type(title),
                    service_type="general",
                    symptoms=[title],
                    investigation_steps=steps[:8],
                    common_causes=self._extract_causes(content),
                    resolution_steps=[],
                    escalation_criteria="",
                )
                entries.append(entry)

        return entries

    def _classify_alert_type(self, title: str) -> str:
        """Classify a section title into an alert type."""
        t = title.lower()
        if any(w in t for w in ["error", "5xx", "failure"]):
            return "HighErrorRate"
        if any(w in t for w in ["latency", "slow", "timeout"]):
            return "HighLatency"
        if any(w in t for w in ["memory", "oom", "heap"]):
            return "OOMKilled"
        if any(w in t for w in ["disk", "storage", "space"]):
            return "DiskFull"
        if any(w in t for w in ["cpu", "load", "throttl"]):
            return "HighCPU"
        return "GeneralIncident"

    def _extract_causes(self, text: str) -> list[str]:
        """Extract cause statements from text."""
        causes = []
        # Look for "because", "due to", "caused by" patterns
        for m in re.finditer(r"(?:because|due to|caused by|result of)\s+(.+?)(?:[.,;]|$)", text, re.IGNORECASE):
            cause = m.group(1).strip()[:150]
            if cause:
                causes.append(cause)
        return causes[:5]

    def _save(self, entries: list[RunbookEntry]) -> None:
        out_path = self.output_dir / "runbooks.jsonl"
        with open(out_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(asdict(entry)) + "\n")
        logger.info(f"Saved {len(entries)} runbook entries to {out_path}")


if __name__ == "__main__":
    corpus = PagerDutyCorpus()
    asyncio.run(corpus.collect())
