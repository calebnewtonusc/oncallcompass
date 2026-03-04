"""
postmortem_crawler.py - Crawls public postmortems from GitHub and engineering blogs.

Sources:
    - danluu/post-mortems (GitHub)
    - upgundecha/howtheysre (GitHub)
    - Engineering blogs: Netflix, Stripe, Cloudflare, Discord, Atlassian
    - SRE Weekly digest links

Output: JSONL with raw postmortem text, source URL, and extracted metadata.

Usage:
    from discovery.postmortem_crawler import PostmortemCrawler
    crawler = PostmortemCrawler()
    docs = await crawler.crawl_all()
"""

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, AsyncIterator

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

try:
    from github import Github

    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False


@dataclass
class PostmortemDoc:
    """A single raw postmortem document."""

    source_url: str
    title: str
    raw_text: str
    company: str
    has_root_cause: bool
    has_action_items: bool
    doc_id: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.sha256(self.source_url.encode()).hexdigest()[:16]

    @property
    def is_quality(self) -> bool:
        """Check if document meets minimum quality threshold."""
        if not self.has_root_cause:
            return False
        if not self.has_action_items:
            return False
        # Minimum length for a useful postmortem
        return len(self.raw_text) >= 500


BLOG_SOURCES: list[dict[str, Any]] = [
    {
        "company": "netflix",
        "rss_url": "https://netflixtechblog.com/feed",
        "keywords": ["outage", "incident", "postmortem", "reliability", "availability"],
    },
    {
        "company": "cloudflare",
        "rss_url": "https://blog.cloudflare.com/rss/",
        "keywords": ["incident", "outage", "postmortem", "disruption"],
    },
    {
        "company": "discord",
        "rss_url": "https://discord.com/blog/rss.xml",
        "keywords": ["outage", "incident", "availability"],
    },
    {
        "company": "stripe",
        "rss_url": "https://stripe.com/blog/feed.rss",
        "keywords": ["incident", "reliability", "availability"],
    },
    {
        "company": "atlassian",
        "base_url": "https://www.atlassian.com/incident-management/postmortem",
        "rss_url": None,
    },
]

GITHUB_REPOS = [
    ("danluu", "post-mortems"),
    ("upgundecha", "howtheysre"),
    ("mr-mig", "every-programmer-should-know"),
]

QUALITY_SIGNALS = {
    "root_cause": [
        r"root\s+cause",
        r"caused\s+by",
        r"root cause analysis",
        r"what happened",
    ],
    "action_items": [
        r"action\s+items?",
        r"follow.?up",
        r"prevention",
        r"remediation",
        r"to.?do",
        r"tasks?",
    ],
}


class PostmortemCrawler:
    def __init__(self, output_dir: str = "data/raw/postmortems"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.github_token = os.environ.get("GITHUB_TOKEN")
        self._seen_ids: set[str] = set()

    async def crawl_all(self) -> list[PostmortemDoc]:
        """Crawl all sources concurrently."""
        docs: list[PostmortemDoc] = []

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "OncallCompass-Research/1.0"},
        ) as session:
            tasks = [
                self._crawl_blogs(session),
                self._crawl_github_repos(),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, BaseException):
                logger.error(f"Crawl error: {result}")
            elif isinstance(result, list):
                docs.extend(result)

        # Deduplicate
        seen = set()
        unique = []
        for doc in docs:
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                unique.append(doc)

        quality = [d for d in unique if d.is_quality]
        logger.info(
            f"Crawled {len(unique)} docs, {len(quality)} meet quality threshold"
        )

        self._save(quality)
        return quality

    async def _crawl_blogs(self, session: aiohttp.ClientSession) -> list[PostmortemDoc]:
        """Crawl RSS feeds from engineering blogs."""
        docs = []
        for source in BLOG_SOURCES:
            if not source.get("rss_url"):
                continue
            try:
                async for doc in self._parse_rss_feed(session, source):
                    docs.append(doc)
            except Exception as e:
                logger.warning(f"Failed to crawl {source['company']}: {e}")
        return docs

    async def _parse_rss_feed(
        self, session: aiohttp.ClientSession, source: dict
    ) -> AsyncIterator[PostmortemDoc]:
        """Parse RSS feed and yield relevant postmortem documents."""
        try:
            async with session.get(source["rss_url"]) as resp:
                text = await resp.text()
        except Exception as e:
            logger.debug(f"RSS fetch failed for {source['company']}: {e}")
            return

        try:
            import feedparser

            feed = feedparser.parse(text)
        except ImportError:
            soup = BeautifulSoup(text, "xml")
            items = soup.find_all("item")
            for item in items[:20]:
                title = item.find("title")
                link = item.find("link")
                desc = item.find("description")
                if title and link:
                    keywords = source.get("keywords", [])
                    title_text = title.get_text().lower()
                    if any(kw in title_text for kw in keywords):
                        doc_text = desc.get_text() if desc else title.get_text()
                        doc = self._build_doc(
                            source_url=link.get_text(),
                            title=title.get_text(),
                            text=doc_text,
                            company=source["company"],
                        )
                        yield doc
            return

        keywords = source.get("keywords", [])
        for entry in feed.entries[:50]:
            entry_title: str = getattr(entry, "title", "").lower()
            if not any(kw in entry_title for kw in keywords):
                continue

            content = (
                getattr(entry, "content", [{}])[0].get("value", "")
                if hasattr(entry, "content")
                else ""
            )
            if not content:
                content = getattr(entry, "summary", "")

            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(separator="\n", strip=True)

            doc = self._build_doc(
                source_url=entry.link,
                title=entry.title,
                text=text,
                company=source["company"],
            )
            yield doc

    async def _crawl_github_repos(self) -> list[PostmortemDoc]:
        """Crawl GitHub repos with postmortem collections."""
        if not HAS_GITHUB:
            logger.warning("PyGithub not installed — skipping GitHub crawl")
            return []

        docs = []
        g = Github(self.github_token) if self.github_token else Github()

        for owner, repo_name in GITHUB_REPOS:
            try:
                repo = g.get_repo(f"{owner}/{repo_name}")
                contents = repo.get_contents("")

                while contents:
                    item = contents.pop(0)
                    if item.type == "dir":
                        contents.extend(repo.get_contents(item.path))
                    elif item.name.endswith(".md") and any(
                        kw in item.name.lower()
                        for kw in ["postmortem", "incident", "outage", "failure"]
                    ):
                        text = item.decoded_content.decode("utf-8", errors="ignore")
                        doc = self._build_doc(
                            source_url=item.html_url,
                            title=item.name,
                            text=text,
                            company=owner,
                        )
                        docs.append(doc)

            except Exception as e:
                logger.warning(f"GitHub repo {owner}/{repo_name} failed: {e}")

        return docs

    def _build_doc(
        self, source_url: str, title: str, text: str, company: str
    ) -> PostmortemDoc:
        """Build a PostmortemDoc from raw text."""
        has_root_cause = any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in QUALITY_SIGNALS["root_cause"]
        )
        has_action_items = any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in QUALITY_SIGNALS["action_items"]
        )
        return PostmortemDoc(
            source_url=source_url,
            title=title,
            raw_text=text[:10000],
            company=company,
            has_root_cause=has_root_cause,
            has_action_items=has_action_items,
        )

    def _save(self, docs: list[PostmortemDoc]) -> None:
        """Save postmortem docs to JSONL."""
        out_path = self.output_dir / "postmortems.jsonl"
        with open(out_path, "w") as f:
            for doc in docs:
                f.write(json.dumps(asdict(doc)) + "\n")
        logger.info(f"Saved {len(docs)} postmortems to {out_path}")


if __name__ == "__main__":
    crawler = PostmortemCrawler()
    asyncio.run(crawler.crawl_all())
