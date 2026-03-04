"""
discovery/sre_weekly.py - Scrape SRE Weekly newsletter archive for incident links.

SRE Weekly (https://sreweekly.com/) is a curated newsletter that links to postmortems,
incident reports, and reliability engineering articles from across the web.
Each issue typically contains 5-20 links, many pointing to real postmortems.

Strategy:
  1. Fetch the archive index to get all issue URLs
  2. For each issue, extract all outbound links and their context
  3. Filter for links that point to postmortem/incident content
  4. Fetch and parse each linked article
  5. Emit (source_url, title, raw_text, company) records compatible with PostmortemDoc

Usage:
    python discovery/sre_weekly.py --output data/raw/sre_weekly
    python discovery/sre_weekly.py --max-issues 50 --output data/raw/sre_weekly
"""

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

SRE_WEEKLY_BASE = "https://sreweekly.com"
SRE_WEEKLY_ARCHIVE = "https://sreweekly.com/category/issues/"

# Link patterns that strongly suggest postmortem/incident content
INCIDENT_URL_PATTERNS = [
    r"postmortem",
    r"incident.report",
    r"outage",
    r"reliability.report",
    r"incident-analysis",
    r"retrospective",
    r"failure",
    r"disruption",
    r"status\.io",
    r"statuspage",
    r"root.cause",
]

# Context keywords around a link that signal it's an incident link
INCIDENT_CONTEXT_KEYWORDS = [
    "postmortem",
    "incident",
    "outage",
    "downtime",
    "root cause",
    "retrospective",
    "failure analysis",
    "disruption",
    "reliability",
    "what happened",
    "investigation",
    "blameless",
    "5 whys",
]

# Domains that frequently publish postmortems
POSTMORTEM_DOMAINS = {
    "netflixtechblog.com",
    "medium.com",
    "engineering.fb.com",
    "blog.cloudflare.com",
    "stripe.com",
    "discord.com",
    "blog.github.com",
    "engineering.linkedin.com",
    "engineering.atspotify.com",
    "eng.lyft.com",
    "slack.engineering",
    "shopify.engineering",
    "github.blog",
    "about.gitlab.com",
    "pagerduty.com",
    "honeycomb.io",
}

QUALITY_PATTERNS = {
    "root_cause": [
        r"root\s+cause",
        r"caused\s+by",
        r"root cause analysis",
        r"what happened",
        r"identified.*cause",
    ],
    "action_items": [
        r"action\s+items?",
        r"follow.?up",
        r"prevention",
        r"remediation",
        r"to.?do",
        r"corrective",
        r"going\s+forward",
    ],
}


@dataclass
class SREWeeklyArticle:
    """An article linked from SRE Weekly."""

    source_url: str
    title: str
    raw_text: str
    company: str
    sre_weekly_issue: int
    has_root_cause: bool
    has_action_items: bool
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = hashlib.sha256(self.source_url.encode()).hexdigest()[:16]

    @property
    def is_quality(self) -> bool:
        return (
            self.has_root_cause and self.has_action_items and len(self.raw_text) >= 400
        )


def _guess_company(url: str, title: str, text: str) -> str:
    """Infer company/author from URL or article content."""
    domain = urlparse(url).netloc.lower()
    # Strip www./blog./engineering. prefixes
    for prefix in ("www.", "blog.", "engineering.", "eng.", "medium.com/"):
        domain = domain.replace(prefix, "")

    if domain:
        return domain.split(".")[0]
    return "unknown"


def _is_incident_link(url: str, link_text: str, surrounding_text: str) -> bool:
    """Determine if a link points to incident/postmortem content."""
    url_lower = url.lower()
    if any(re.search(p, url_lower) for p in INCIDENT_URL_PATTERNS):
        return True

    combined_text = f"{link_text} {surrounding_text}".lower()
    if any(kw in combined_text for kw in INCIDENT_CONTEXT_KEYWORDS):
        return True

    domain = urlparse(url).netloc.lower().replace("www.", "")
    if any(pm_domain in domain for pm_domain in POSTMORTEM_DOMAINS):
        combined = f"{link_text} {surrounding_text}".lower()
        if any(
            kw in combined
            for kw in ["incident", "outage", "downtime", "postmortem", "failure"]
        ):
            return True

    return False


def _check_quality(text: str) -> tuple[bool, bool]:
    """Check if text has root cause and action items signals."""
    has_root = any(
        re.search(p, text, re.IGNORECASE) for p in QUALITY_PATTERNS["root_cause"]
    )
    has_actions = any(
        re.search(p, text, re.IGNORECASE) for p in QUALITY_PATTERNS["action_items"]
    )
    return has_root, has_actions


async def _fetch_text(
    session: aiohttp.ClientSession,
    url: str,
    timeout: int = 20,
) -> Optional[str]:
    """Fetch URL and return text content."""
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=True,
        ) as resp:
            if resp.status != 200:
                return None
            return await resp.text(errors="ignore")
    except Exception as e:
        logger.debug(f"Fetch failed for {url}: {e}")
        return None


async def _get_issue_urls(
    session: aiohttp.ClientSession, max_pages: int = 20
) -> list[str]:
    """Crawl the SRE Weekly archive and collect individual issue URLs."""
    issue_urls: list[str] = []
    page = 1

    while page <= max_pages:
        archive_url = (
            f"{SRE_WEEKLY_ARCHIVE}page/{page}/" if page > 1 else SRE_WEEKLY_ARCHIVE
        )
        html = await _fetch_text(session, archive_url)
        if not html:
            break

        soup = BeautifulSoup(html, "html.parser")

        # SRE Weekly uses standard WordPress-style archive page
        links_found = 0
        for a_tag in soup.find_all("a", href=re.compile(r"/sre-weekly-issue-\d+")):
            raw_href = a_tag["href"]
            href = urljoin(SRE_WEEKLY_BASE, str(raw_href) if isinstance(raw_href, list) else raw_href)
            if href not in issue_urls:
                issue_urls.append(href)
                links_found += 1

        # Also try h2/h3 article title links
        for heading in soup.find_all(["h2", "h3"]):
            a = heading.find("a")
            if a and a.get("href"):
                raw_a_href = a["href"]
                href = urljoin(SRE_WEEKLY_BASE, str(raw_a_href) if isinstance(raw_a_href, list) else raw_a_href)
                if "sre-weekly" in href.lower() and href not in issue_urls:
                    issue_urls.append(href)
                    links_found += 1

        if links_found == 0:
            break

        logger.debug(
            f"Archive page {page}: found {links_found} issues (total: {len(issue_urls)})"
        )
        page += 1

    return issue_urls


async def _extract_links_from_issue(
    session: aiohttp.ClientSession,
    issue_url: str,
    issue_number: int,
) -> list[dict]:
    """Extract outbound incident links from one SRE Weekly issue."""
    html = await _fetch_text(session, issue_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    content_div = soup.find(
        "div", class_=re.compile("entry-content|post-content|article-content")
    )
    if not content_div:
        content_div = soup.find("article") or soup

    candidate_links = []
    for a_tag in content_div.find_all("a", href=True):
        raw_href = a_tag["href"]
        href: str = (
            " ".join(raw_href) if isinstance(raw_href, list) else str(raw_href)
        )
        if not href.startswith("http"):
            continue
        if "sreweekly.com" in href:
            continue

        link_text = a_tag.get_text(strip=True)

        # Get surrounding paragraph context
        parent = a_tag.parent
        surrounding = parent.get_text(separator=" ", strip=True)[:300] if parent else ""

        if _is_incident_link(href, link_text, surrounding):
            candidate_links.append(
                {
                    "url": href,
                    "link_text": link_text,
                    "context": surrounding,
                    "issue_number": issue_number,
                }
            )

    return candidate_links


async def _fetch_and_parse_article(
    session: aiohttp.ClientSession,
    link_info: dict,
) -> Optional[SREWeeklyArticle]:
    """Fetch a linked article and parse it into SREWeeklyArticle."""
    url = link_info["url"]
    html = await _fetch_text(session, url, timeout=25)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_tag = soup.find("h1") or soup.find("title")
    title = (
        title_tag.get_text(strip=True) if title_tag else link_info.get("link_text", "")
    )

    # Extract main text content
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Prefer article/main content
    content = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile("content|article|post"))
    )
    if content:
        raw_text = content.get_text(separator="\n", strip=True)
    else:
        raw_text = soup.get_text(separator="\n", strip=True)

    # Truncate to reasonable length
    raw_text = raw_text[:12000]

    if len(raw_text) < 200:
        return None

    has_root, has_actions = _check_quality(raw_text)
    company = _guess_company(url, title, raw_text)

    return SREWeeklyArticle(
        source_url=url,
        title=title[:200],
        raw_text=raw_text,
        company=company,
        sre_weekly_issue=link_info.get("issue_number", 0),
        has_root_cause=has_root,
        has_action_items=has_actions,
    )


async def crawl_sre_weekly(
    output_dir: str = "data/raw/sre_weekly",
    max_issues: int = 300,
    concurrency: int = 10,
) -> list[SREWeeklyArticle]:
    """Main crawl pipeline: SRE Weekly archive -> incident article corpus."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    existing_file = out_path / "sre_weekly_articles.jsonl"
    if existing_file.exists():
        with open(existing_file) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen_ids.add(rec.get("doc_id", ""))
                except json.JSONDecodeError:
                    pass
        logger.info(f"Loaded {len(seen_ids)} existing doc IDs")

    async with aiohttp.ClientSession(
        headers={
            "User-Agent": "OncallCompass-Research/1.0 (postmortem corpus crawler)"
        },
    ) as session:
        logger.info("Fetching SRE Weekly archive index...")
        issue_urls = await _get_issue_urls(session, max_pages=max_issues // 10 + 5)
        issue_urls = issue_urls[:max_issues]
        logger.info(f"Found {len(issue_urls)} issues to process")

        # Extract all candidate links from all issues
        all_links: list[dict] = []
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_issue(url: str, num: int) -> list[dict]:
            async with semaphore:
                return await _extract_links_from_issue(session, url, num)

        link_tasks = [bounded_issue(url, i + 1) for i, url in enumerate(issue_urls)]
        link_results = await asyncio.gather(*link_tasks, return_exceptions=True)
        for result in link_results:
            if isinstance(result, list):
                all_links.extend(result)

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_links = []
        for link in all_links:
            if link["url"] not in seen_urls:
                seen_urls.add(link["url"])
                unique_links.append(link)

        logger.info(f"Found {len(unique_links)} unique candidate incident links")

        # Fetch and parse each article
        articles: list[SREWeeklyArticle] = []

        async def bounded_article(link: dict) -> Optional[SREWeeklyArticle]:
            async with semaphore:
                return await _fetch_and_parse_article(session, link)

        article_tasks = [bounded_article(link) for link in unique_links]
        article_results = await asyncio.gather(*article_tasks, return_exceptions=True)

        for result in article_results:
            if isinstance(result, SREWeeklyArticle):
                if result.doc_id not in seen_ids and result.is_quality:
                    articles.append(result)
                    seen_ids.add(result.doc_id)

    logger.info(f"Collected {len(articles)} quality articles")

    # Save — write all new articles in one sequential append to reduce the
    # window for concurrent write races. True cross-process locking would
    # require fcntl/msvcrt, but single-process usage is the expected case.
    if articles:
        with open(existing_file, "a") as f:
            for article in articles:
                f.write(json.dumps(asdict(article)) + "\n")

    logger.info(f"Saved {len(articles)} articles to {existing_file}")
    return articles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Crawl SRE Weekly for incident articles"
    )
    parser.add_argument("--output", default="data/raw/sre_weekly")
    parser.add_argument(
        "--max-issues",
        type=int,
        default=300,
        help="Maximum number of SRE Weekly issues to process",
    )
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    articles = asyncio.run(
        crawl_sre_weekly(
            output_dir=args.output,
            max_issues=args.max_issues,
            concurrency=args.concurrency,
        )
    )
    logger.info(f"Done. {len(articles)} articles saved to {args.output}")
