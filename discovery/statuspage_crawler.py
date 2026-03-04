"""
discovery/statuspage_crawler.py - Crawls public Statuspage.io incident APIs.

Sources:
    - GitHub Status (githubstatus.com)
    - Cloudflare Status (cloudflarestatus.com)
    - Stripe Status (status.stripe.com)
    - Twilio Status (status.twilio.com)
    - Heroku Status (status.heroku.com)
    - PagerDuty Status (status.pagerduty.com)
    - Datadog Status (status.datadoghq.com)
    - Atlassian Status (jira-software.status.atlassian.com)
    - Shopify Status (www.shopifystatus.com)
    - Sendgrid Status (status.sendgrid.com)
    - Fastly Status (status.fastly.com)
    - Zendesk Status (status.zendesk.com)
    - CircleCI Status (status.circleci.com)

Creates (incident_update_stream, root_cause, resolution) records.

Usage:
    python discovery/statuspage_crawler.py --output data/raw/statuspages
"""

import asyncio
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
from loguru import logger


@dataclass
class StatuspageIncident:
    """A resolved incident from a Statuspage.io API."""

    provider: str
    incident_id: str
    title: str
    impact: str  # none|minor|major|critical
    status: str  # resolved
    created_at: str
    resolved_at: str
    duration_minutes: int
    affected_components: list[str]
    update_stream: list[dict]  # [{time, status, body}] chronological
    root_cause: str
    resolution_summary: str
    has_root_cause: bool
    incident_url: str


# All known Statuspage.io-compatible public APIs
STATUSPAGE_SOURCES = [
    {"provider": "github", "url": "https://www.githubstatus.com/api/v2/incidents.json"},
    {
        "provider": "cloudflare",
        "url": "https://www.cloudflarestatus.com/api/v2/incidents.json",
    },
    {"provider": "stripe", "url": "https://status.stripe.com/api/v2/incidents.json"},
    {"provider": "twilio", "url": "https://status.twilio.com/api/v2/incidents.json"},
    {"provider": "heroku", "url": "https://status.heroku.com/api/v2/incidents.json"},
    {
        "provider": "pagerduty",
        "url": "https://status.pagerduty.com/api/v2/incidents.json",
    },
    {
        "provider": "datadog",
        "url": "https://status.datadoghq.com/api/v2/incidents.json",
    },
    {
        "provider": "atlassian",
        "url": "https://jira-software.status.atlassian.com/api/v2/incidents.json",
    },
    {
        "provider": "shopify",
        "url": "https://www.shopifystatus.com/api/v2/incidents.json",
    },
    {
        "provider": "sendgrid",
        "url": "https://status.sendgrid.com/api/v2/incidents.json",
    },
    {"provider": "fastly", "url": "https://status.fastly.com/api/v2/incidents.json"},
    {"provider": "zendesk", "url": "https://status.zendesk.com/api/v2/incidents.json"},
    {
        "provider": "circleci",
        "url": "https://status.circleci.com/api/v2/incidents.json",
    },
    {"provider": "npm", "url": "https://www.npmjs.com/status/api/v2/incidents.json"},
    {
        "provider": "digitalocean",
        "url": "https://status.digitalocean.com/api/v2/incidents.json",
    },
    {"provider": "discord", "url": "https://discordstatus.com/api/v2/incidents.json"},
    {
        "provider": "notion",
        "url": "https://www.notionstatuspage.com/api/v2/incidents.json",
    },
    {
        "provider": "vercel",
        "url": "https://www.vercel-status.com/api/v2/incidents.json",
    },
    {
        "provider": "confluent",
        "url": "https://status.confluent.cloud/api/v2/incidents.json",
    },
    {"provider": "elastic", "url": "https://status.elastic.co/api/v2/incidents.json"},
]

ROOT_CAUSE_PATTERNS = [
    r"root\s+cause[:\s]+(.*?)(?:[\.\n]|$)",
    r"caused\s+by[:\s]+(.*?)(?:[\.\n]|$)",
    r"the\s+issue\s+was\s+(?:due\s+to|caused\s+by)[:\s]+(.*?)(?:[\.\n]|$)",
    r"identified\s+(?:the\s+)?(?:root\s+)?cause(?:\s+as)?[:\s]+(.*?)(?:[\.\n]|$)",
    r"the\s+issue\s+was[:\s]+(.*?)(?:[\.\n]|$)",
    r"we\s+identified[:\s]+(.*?)(?:[\.\n]|$)",
    r"this\s+was\s+caused\s+by[:\s]+(.*?)(?:[\.\n]|$)",
    r"traced\s+(?:this\s+)?(?:back\s+)?to[:\s]+(.*?)(?:[\.\n]|$)",
]


def _parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse ISO 8601 datetime strings, handling variants with/without microseconds."""
    if not dt_str:
        return None
    # Normalize offset variants: replace +00:00 with Z for uniform handling.
    normalized = dt_str.strip().replace("+00:00", "Z").replace(" ", "T")
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",  # with microseconds + Z
        "%Y-%m-%dT%H:%M:%SZ",  # without microseconds + Z
        "%Y-%m-%dT%H:%M:%S.%f",  # with microseconds, no suffix
        "%Y-%m-%dT%H:%M:%S",  # bare datetime
    ):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def _compute_duration(start: str, end: str) -> int:
    s = _parse_datetime(start)
    e = _parse_datetime(end)
    if s and e:
        return max(0, int((e - s).total_seconds() / 60))
    return 0


def _extract_root_cause(text: str) -> str:
    for pattern in ROOT_CAUSE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            candidate = m.group(1).strip()[:400]
            if len(candidate) > 20:
                return candidate
    # Fallback: take first sentence of resolution text
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if sentences and len(sentences[0]) > 20:
        return sentences[0][:400]
    return ""


def _parse_incident(raw: dict, provider: str) -> Optional[StatuspageIncident]:
    """Parse a raw Statuspage.io incident JSON into StatuspageIncident."""
    try:
        if raw.get("status") != "resolved":
            return None

        updates = sorted(
            raw.get("incident_updates", []),
            key=lambda u: u.get("created_at", ""),
        )

        update_stream = [
            {
                "time": u.get("created_at", ""),
                "status": u.get("status", ""),
                "body": u.get("body", "")[:600],
            }
            for u in updates
        ]

        # Gather resolution text (last few updates for richer root cause extraction)
        resolution_text = " ".join(u.get("body", "") for u in updates[-3:])
        root_cause = _extract_root_cause(resolution_text)

        resolution_update = next(
            (u for u in reversed(updates) if u.get("status") == "resolved"), None
        )
        resolution_summary = (
            resolution_update["body"][:800] if resolution_update else ""
        )

        affected = [
            c.get("name", "")
            for c in raw.get("components", [])
            if c.get("status") not in ("operational", "")
        ]

        start = raw.get("created_at", "")
        end = raw.get("resolved_at", "")
        duration = _compute_duration(start, end)

        shortlink = raw.get("shortlink", "")
        if not shortlink:
            shortlink = f"https://{provider}status.com/incidents/{raw.get('id', '')}"

        return StatuspageIncident(
            provider=provider,
            incident_id=raw.get("id", ""),
            title=raw.get("name", ""),
            impact=raw.get("impact", "none"),
            status="resolved",
            created_at=start,
            resolved_at=end,
            duration_minutes=duration,
            affected_components=affected,
            update_stream=update_stream,
            root_cause=root_cause,
            resolution_summary=resolution_summary,
            has_root_cause=len(root_cause) > 20,
            incident_url=shortlink,
        )
    except Exception as e:
        logger.debug(f"Failed to parse incident from {provider}: {e}")
        return None


async def _fetch_incidents(
    session: aiohttp.ClientSession,
    source: dict,
) -> list[StatuspageIncident]:
    """Fetch and parse all incidents from one Statuspage.io API endpoint."""
    provider = source["provider"]
    url = source["url"]
    incidents: list[StatuspageIncident] = []

    # Statuspage.io supports pagination via ?page= for older incidents
    for page in range(0, 10):
        page_url = f"{url}?page={page}" if page > 0 else url
        try:
            async with session.get(
                page_url,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 404:
                    break
                if resp.status != 200:
                    logger.debug(f"{provider} API returned {resp.status}")
                    break
                data = await resp.json(content_type=None)
        except Exception as e:
            logger.debug(f"Failed to fetch {provider} page {page}: {e}")
            break

        raw_incidents = data.get("incidents", [])
        if not raw_incidents:
            break

        for raw in raw_incidents:
            incident = _parse_incident(raw, provider)
            if incident:
                incidents.append(incident)

        # Statuspage.io returns up to 100 per page; fewer means last page
        if len(raw_incidents) < 100:
            break

    logger.info(f"  {provider}: {len(incidents)} resolved incidents")
    return incidents


async def crawl_all_statuspages(
    output_dir: str = "data/raw/statuspages",
) -> list[StatuspageIncident]:
    """Crawl all configured Statuspage.io endpoints concurrently."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession(
        headers={"User-Agent": "OncallCompass-Research/1.0"},
    ) as session:
        tasks = [_fetch_incidents(session, src) for src in STATUSPAGE_SOURCES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_incidents: list[StatuspageIncident] = []
    for src, result in zip(STATUSPAGE_SOURCES, results):
        if isinstance(result, Exception):
            logger.warning(f"Error crawling {src['provider']}: {result}")
        else:
            all_incidents.extend(result)

    # Quality filter: must have root cause and duration
    quality = [
        i
        for i in all_incidents
        if i.has_root_cause and i.duration_minutes > 0 and len(i.update_stream) >= 2
    ]

    logger.info(
        f"Total incidents: {len(all_incidents)}, quality: {len(quality)} "
        f"(with root cause + duration + ≥2 updates)"
    )

    # Deduplicate by incident_id
    seen_ids: set[str] = set()
    deduped = []
    for inc in quality:
        key = f"{inc.provider}:{inc.incident_id}"
        if key not in seen_ids:
            seen_ids.add(key)
            deduped.append(inc)

    # Save all
    all_path = out_path / "statuspage_incidents.jsonl"
    with open(all_path, "w") as f:
        for inc in deduped:
            f.write(json.dumps(asdict(inc)) + "\n")
    logger.info(f"Saved {len(deduped)} incidents to {all_path}")

    # Save per-provider breakdown
    by_provider: dict[str, list] = {}
    for inc in deduped:
        by_provider.setdefault(inc.provider, []).append(inc)

    for provider, incs in by_provider.items():
        ppath = out_path / f"{provider}_incidents.jsonl"
        with open(ppath, "w") as f:
            for inc in incs:
                f.write(json.dumps(asdict(inc)) + "\n")

    # Emit summary
    summary = {p: len(v) for p, v in by_provider.items()}
    summary["_total"] = len(deduped)
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Provider breakdown: "
        + ", ".join(f"{p}={n}" for p, n in sorted(summary.items()))
    )
    return deduped


def build_training_records(incidents: list[StatuspageIncident]) -> list[dict]:
    """
    Convert raw incidents into (incident_update_stream, root_cause, resolution) records
    for training pair synthesis.
    """
    records = []
    for inc in incidents:
        if not inc.has_root_cause or not inc.update_stream:
            continue

        # Format the update stream as a narrative
        stream_text = "\n".join(
            f"[{u['time'][:16]}] [{u['status'].upper()}] {u['body']}"
            for u in inc.update_stream
        )

        records.append(
            {
                "provider": inc.provider,
                "incident_id": inc.incident_id,
                "title": inc.title,
                "impact": inc.impact,
                "duration_minutes": inc.duration_minutes,
                "affected_components": inc.affected_components,
                "incident_update_stream": stream_text,
                "root_cause": inc.root_cause,
                "resolution": inc.resolution_summary,
                "incident_url": inc.incident_url,
            }
        )

    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crawl Statuspage.io incident APIs")
    parser.add_argument("--output", default="data/raw/statuspages")
    parser.add_argument(
        "--training-records",
        action="store_true",
        help="Also output formatted training records",
    )
    args = parser.parse_args()

    incidents = asyncio.run(crawl_all_statuspages(output_dir=args.output))

    if args.training_records:
        records = build_training_records(incidents)
        records_path = Path(args.output) / "training_records.jsonl"
        with open(records_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        logger.info(f"Saved {len(records)} training records to {records_path}")
