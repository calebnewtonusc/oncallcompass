"""
aws_incidents.py - Harvests incidents from AWS, GCP, and Azure status pages.

Sources:
    - AWS Health Dashboard (health.aws.amazon.com)
    - Google Cloud Status (status.cloud.google.com)
    - Azure Status (status.azure.com)
    - Cloudflare System Status (www.cloudflarestatus.com)

Each status page incident becomes a training example with:
    - Affected services (mapped to generic service categories)
    - Incident timeline (start → investigation → resolution)
    - Root cause (from final resolution note)
    - Mitigation steps

Usage:
    from discovery.aws_incidents import CloudStatusCrawler
    crawler = CloudStatusCrawler()
    incidents = await crawler.crawl_all()
"""

import asyncio
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp
from loguru import logger


@dataclass
class CloudIncident:
    """A cloud provider status page incident."""

    provider: str  # "aws" | "gcp" | "azure" | "cloudflare"
    incident_id: str
    title: str
    affected_services: list[str]
    severity: str  # "minor" | "major" | "critical"
    start_time: str
    end_time: str
    duration_minutes: int
    timeline: list[dict]  # [{time, status, description}]
    root_cause: str
    resolution: str
    regions: list[str]


STATUS_APIS = {
    "cloudflare": "https://www.cloudflarestatus.com/api/v2/incidents.json",
    "github": "https://www.githubstatus.com/api/v2/incidents.json",
    "atlassian": "https://jira-software.status.atlassian.com/api/v2/incidents.json",
    "pagerduty": "https://status.pagerduty.com/api/v2/incidents.json",
    "datadog": "https://status.datadoghq.com/api/v2/incidents.json",
}


class CloudStatusCrawler:
    def __init__(self, output_dir: str = "data/raw/cloud_incidents"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def crawl_all(self) -> list[CloudIncident]:
        """Crawl all configured status page APIs."""
        all_incidents: list[CloudIncident] = []

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "OncallCompass-Research/1.0"},
        ) as session:
            tasks = [
                self._crawl_statuspage_api(session, provider, url)
                for provider, url in STATUS_APIS.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for provider, result in zip(STATUS_APIS.keys(), results):
            if isinstance(result, BaseException):
                logger.warning(f"Failed to crawl {provider}: {result}")
            elif isinstance(result, list):
                all_incidents.extend(result)
                logger.info(f"Crawled {len(result)} incidents from {provider}")

        # Filter for incidents with root cause information
        quality = [i for i in all_incidents if self._has_root_cause(i)]
        logger.info(
            f"Total: {len(all_incidents)} incidents, {len(quality)} with root cause"
        )

        self._save(quality)
        return quality

    async def _crawl_statuspage_api(
        self, session: aiohttp.ClientSession, provider: str, url: str
    ) -> list[CloudIncident]:
        """Crawl a Statuspage.io-compatible API endpoint."""
        incidents = []
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception as e:
            logger.debug(f"API call failed for {provider}: {e}")
            return []

        raw_incidents = data.get("incidents", [])

        for raw in raw_incidents:
            # Only resolved incidents have root cause information
            if raw.get("status") != "resolved":
                continue

            incident = self._parse_statuspage_incident(raw, provider)
            if incident:
                incidents.append(incident)

        return incidents

    def _parse_statuspage_incident(
        self, raw: dict, provider: str
    ) -> CloudIncident | None:
        """Parse a Statuspage.io incident JSON into CloudIncident."""
        try:
            updates = raw.get("incident_updates", [])
            # Sort by created_at descending (newest first)
            updates = sorted(
                updates, key=lambda u: u.get("created_at", ""), reverse=True
            )

            # Build timeline from updates
            timeline = []
            for update in reversed(updates):
                timeline.append(
                    {
                        "time": update.get("created_at", ""),
                        "status": update.get("status", ""),
                        "description": update.get("body", "")[:500],
                    }
                )

            # Extract root cause from resolution update
            resolution_update = next(
                (u for u in updates if u.get("status") == "resolved"), None
            )
            root_cause = ""
            resolution = ""
            if resolution_update:
                body = resolution_update.get("body", "")
                root_cause = self._extract_root_cause(body)
                resolution = body[:1000]

            # Get affected components
            affected = [
                c.get("name", "")
                for c in raw.get("components", [])
                if c.get("status") not in ("operational", "")
            ]

            # Estimate duration
            start = raw.get("created_at", "")
            end = raw.get("resolved_at", "")
            duration = 0
            if start and end:
                try:
                    from datetime import datetime

                    # Try multiple ISO 8601 variants; some providers omit microseconds.
                    _formats = [
                        "%Y-%m-%dT%H:%M:%S.%fZ",
                        "%Y-%m-%dT%H:%M:%SZ",
                        "%Y-%m-%dT%H:%M:%S.%f+00:00",
                        "%Y-%m-%dT%H:%M:%S+00:00",
                    ]

                    def _parse(dt_str: str):
                        for _fmt in _formats:
                            try:
                                return datetime.strptime(
                                    dt_str[:26],
                                    _fmt.rstrip("Z")
                                    + ("Z" if _fmt.endswith("Z") else ""),
                                )
                            except ValueError:
                                pass
                        for _fmt in _formats:
                            try:
                                return datetime.strptime(dt_str, _fmt)
                            except ValueError:
                                pass
                        return None

                    s = _parse(start)
                    e = _parse(end)
                    if s and e:
                        duration = int((e - s).total_seconds() / 60)
                except Exception:
                    pass

            return CloudIncident(
                provider=provider,
                incident_id=raw.get("id", ""),
                title=raw.get("name", ""),
                affected_services=affected,
                severity=raw.get("impact", "minor"),
                start_time=start,
                end_time=end,
                duration_minutes=duration,
                timeline=timeline,
                root_cause=root_cause,
                resolution=resolution,
                regions=[],
            )

        except Exception as e:
            logger.debug(f"Failed to parse incident: {e}")
            return None

    def _extract_root_cause(self, text: str) -> str:
        """Extract root cause sentence from resolution update text."""
        # Look for explicit root cause statement
        patterns = [
            r"root\s+cause[:\s]+(.*?)(?:\.|$)",
            r"caused\s+by[:\s]+(.*?)(?:\.|$)",
            r"identified[:\s]+(.*?)\s+as\s+the\s+cause",
            r"the\s+issue\s+was[:\s]+(.*?)(?:\.|$)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).strip()[:300]

        # Fallback: first sentence of resolution
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return sentences[0][:300] if sentences else ""

    def _has_root_cause(self, incident: CloudIncident) -> bool:
        """Check if incident has meaningful root cause information."""
        return (
            len(incident.root_cause) > 30
            and incident.duration_minutes > 0
            and len(incident.affected_services) > 0
        )

    def _save(self, incidents: list[CloudIncident]) -> None:
        """Save incidents to JSONL."""
        out_path = self.output_dir / "cloud_incidents.jsonl"
        with open(out_path, "w") as f:
            for incident in incidents:
                f.write(json.dumps(asdict(incident)) + "\n")
        logger.info(f"Saved {len(incidents)} cloud incidents to {out_path}")


if __name__ == "__main__":
    crawler = CloudStatusCrawler()
    asyncio.run(crawler.crawl_all())
