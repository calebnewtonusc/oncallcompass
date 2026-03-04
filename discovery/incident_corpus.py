"""
OncallCompass Incident Corpus Crawler.

Crawls public postmortems and runbooks from:
    - GitHub repositories (postmortem repos, SRE wikis)
    - Engineering blogs (Cloudflare, Netflix, GitLab, Stripe, etc.)
    - AWS/GCP/Azure status page incident histories
    - Public incident ticket exports (PagerDuty, OpsGenie samples)
    - SRE book examples and public runbook collections

Output: data/raw/ directory with normalized incident documents

Usage:
    python discovery/incident_corpus.py \
        --output_dir data/raw \
        --github_token $GITHUB_TOKEN \
        --sources all
"""

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

# Public GitHub repos known to contain postmortems
GITHUB_POSTMORTEM_REPOS = [
    "danluu/post-mortems",
    "mtdvio/every-programmers-security",
    "tldr-sec/public-incident-reports",
]

# Engineering blog postmortem feeds/pages
BLOG_SOURCES = [
    "https://blog.cloudflare.com/tag/outage/",
    "https://netflixtechblog.com/tagged/incidents",
    "https://about.gitlab.com/blog/categories/engineering/",
    "https://stripe.com/blog/engineering",
]

# Public runbook repositories on GitHub
RUNBOOK_REPOS = [
    "SkeltonThatcher/run-book-template",
    "kubernetes/runbooks",
    "grafana/runbooks",
    "monzo/response",
]


@dataclass
class IncidentDocument:
    """A normalized incident document from any source."""
    source: str
    source_url: str
    title: str
    date: str | None
    content: str
    doc_type: str  # "postmortem" | "runbook" | "incident_ticket"
    metadata: dict[str, Any]

    def doc_id(self) -> str:
        return hashlib.sha256(self.source_url.encode()).hexdigest()[:16]


class GitHubPostmortemCrawler:
    """Crawls GitHub repositories for postmortem markdown files."""

    def __init__(self, token: str | None = None) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "OncallCompass-Corpus-Crawler/1.0",
        })
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def crawl_repo(self, repo: str) -> list[IncidentDocument]:
        """Crawl all markdown files from a GitHub repository."""
        docs = []
        url = f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1"
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            tree = resp.json().get("tree", [])
        except requests.RequestException as e:
            console.print(f"[yellow]Failed to crawl {repo}: {e}[/yellow]")
            return docs

        md_files = [
            item for item in tree
            if item.get("type") == "blob" and item.get("path", "").endswith(".md")
        ]
        console.print(f"  Found {len(md_files)} markdown files in {repo}")

        for item in md_files[:50]:  # Cap per repo to avoid rate limits
            raw_url = f"https://raw.githubusercontent.com/{repo}/HEAD/{item['path']}"
            try:
                raw_resp = self.session.get(raw_url, timeout=20)
                raw_resp.raise_for_status()
                content = raw_resp.text

                # Heuristic: skip files that are too short to be a postmortem
                if len(content) < 200:
                    continue

                doc_type = self._classify_doc(item["path"], content)
                docs.append(IncidentDocument(
                    source=f"github:{repo}",
                    source_url=raw_url,
                    title=item["path"],
                    date=None,
                    content=content,
                    doc_type=doc_type,
                    metadata={"repo": repo, "path": item["path"]},
                ))
                time.sleep(0.1)  # Be polite
            except requests.RequestException:
                continue

        return docs

    def _classify_doc(self, path: str, content: str) -> str:
        """Classify a document as postmortem, runbook, or incident ticket."""
        path_lower = path.lower()
        content_lower = content.lower()

        if any(kw in path_lower for kw in ["postmortem", "post-mortem", "incident", "outage", "rca"]):
            return "postmortem"
        if any(kw in path_lower for kw in ["runbook", "run-book", "playbook", "procedure"]):
            return "runbook"
        if any(kw in content_lower[:500] for kw in ["root cause", "timeline", "impact", "resolution"]):
            return "postmortem"
        return "runbook"


class BlogPostmortemCrawler:
    """Crawls engineering blog pages for postmortem articles."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "OncallCompass-Corpus-Crawler/1.0"

    def crawl(self, url: str, max_articles: int = 20) -> list[IncidentDocument]:
        """Crawl a blog listing page and extract incident-related articles."""
        docs = []
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            console.print(f"[yellow]Failed to crawl {url}: {e}[/yellow]")
            return docs

        # Find article links
        links = soup.find_all("a", href=True)
        article_urls = []
        for link in links:
            href = link["href"]
            full_url = urljoin(url, href)
            text = link.get_text(strip=True).lower()
            if any(kw in text for kw in ["incident", "outage", "postmortem", "rca", "failure", "down"]):
                if urlparse(full_url).netloc == urlparse(url).netloc:
                    article_urls.append(full_url)

        console.print(f"  Found {len(article_urls)} candidate articles at {url}")

        for article_url in article_urls[:max_articles]:
            doc = self._fetch_article(article_url)
            if doc:
                docs.append(doc)
            time.sleep(0.5)

        return docs

    def _fetch_article(self, url: str) -> IncidentDocument | None:
        """Fetch and parse a single blog article."""
        try:
            resp = self.session.get(url, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            title = soup.find("h1")
            title_text = title.get_text(strip=True) if title else url

            # Extract main content
            article = soup.find("article") or soup.find("main") or soup.find("body")
            if not article:
                return None

            # Remove nav, footer, sidebar noise
            for tag in article.find_all(["nav", "footer", "aside", "script", "style"]):
                tag.decompose()

            content = article.get_text(separator="\n", strip=True)
            if len(content) < 300:
                return None

            return IncidentDocument(
                source=f"blog:{urlparse(url).netloc}",
                source_url=url,
                title=title_text,
                date=None,
                content=content,
                doc_type="postmortem",
                metadata={"source_url": url},
            )
        except requests.RequestException:
            return None


def save_corpus(docs: list[IncidentDocument], output_dir: Path) -> None:
    """Save crawled documents to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"

    # Load already-known doc IDs to prevent duplicate manifest entries.
    # Opening in append mode for individual runs causes the manifest to accumulate
    # duplicate records across reruns even though individual doc files are deduplicated.
    existing_ids: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path) as _mf:
            for _line in _mf:
                _line = _line.strip()
                if _line:
                    try:
                        existing_ids.add(json.loads(_line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass

    saved = 0
    new_entries: list[str] = []
    for doc in docs:
        doc_id = doc.doc_id()
        doc_path = output_dir / f"{doc.doc_type}_{doc_id}.json"

        if doc_path.exists() or doc_id in existing_ids:
            continue  # Already crawled or already in manifest

        with open(doc_path, "w") as f:
            json.dump(asdict(doc), f, indent=2)

        entry = json.dumps({
            "id": doc_id,
            "source": doc.source,
            "url": doc.source_url,
            "title": doc.title,
            "type": doc.doc_type,
            "path": str(doc_path),
        })
        new_entries.append(entry)
        existing_ids.add(doc_id)
        saved += 1

    # Append only genuinely new entries to the manifest in one write.
    if new_entries:
        with open(manifest_path, "a") as manifest_f:
            for entry in new_entries:
                manifest_f.write(entry + "\n")

    console.print(f"Saved {saved} new documents to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OncallCompass corpus crawler")
    parser.add_argument("--output_dir", default="data/raw")
    parser.add_argument("--github_token", default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument("--sources", choices=["all", "github", "blogs"], default="all")
    parser.add_argument("--max_articles_per_blog", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_docs: list[IncidentDocument] = []

    if args.sources in ("all", "github"):
        console.print("[bold cyan]Crawling GitHub postmortem repositories...[/bold cyan]")
        github_crawler = GitHubPostmortemCrawler(token=args.github_token)

        for repo in GITHUB_POSTMORTEM_REPOS:
            console.print(f"  Crawling {repo}...")
            docs = github_crawler.crawl_repo(repo)
            all_docs.extend(docs)
            console.print(f"  Got {len(docs)} documents from {repo}")

        for repo in RUNBOOK_REPOS:
            console.print(f"  Crawling runbooks from {repo}...")
            docs = github_crawler.crawl_repo(repo)
            all_docs.extend(docs)
            console.print(f"  Got {len(docs)} runbooks from {repo}")

    if args.sources in ("all", "blogs"):
        console.print("[bold cyan]Crawling engineering blogs...[/bold cyan]")
        blog_crawler = BlogPostmortemCrawler()

        for blog_url in BLOG_SOURCES:
            console.print(f"  Crawling {blog_url}...")
            docs = blog_crawler.crawl(blog_url, max_articles=args.max_articles_per_blog)
            all_docs.extend(docs)
            console.print(f"  Got {len(docs)} articles from {blog_url}")

    console.print(f"\n[bold]Total documents crawled: {len(all_docs)}[/bold]")

    by_type = {}
    for doc in all_docs:
        by_type[doc.doc_type] = by_type.get(doc.doc_type, 0) + 1
    for doc_type, count in by_type.items():
        console.print(f"  {doc_type}: {count}")

    save_corpus(all_docs, output_dir)
    console.print("[green]Corpus crawl complete.[/green]")


if __name__ == "__main__":
    main()
