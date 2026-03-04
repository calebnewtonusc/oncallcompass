"""
synthesize_bulk.py - Bulk synthesis pipeline for OncallCompass training data.

Uses vLLM (on-cluster) or Anthropic API (fallback) to synthesize:
  - SFT training pairs from raw postmortems
  - Synthetic incident scenarios
  - DPO preference pairs
  - CompassBench drill scenarios

Dispatches requests across two inference endpoints for throughput.

Usage:
    python synthesis/synthesize_bulk.py \
        --raw_dir data/raw \
        --output_dir data \
        --vllm_endpoint_1 http://localhost:8000 \
        --vllm_endpoint_2 http://localhost:8001
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import asdict
from typing import Any

import aiohttp
from loguru import logger

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from synthesis.prompts import (
    SFT_EXTRACTION_SYSTEM,
    SFT_EXTRACTION_USER_TEMPLATE,
    INCIDENT_SYNTHESIS_SYSTEM,
    INCIDENT_SYNTHESIS_USER_TEMPLATE,
    DPO_SYSTEM,
    DPO_USER_TEMPLATE,
    DRILL_SYNTHESIS_SYSTEM,
)

# ─────────────────────────────────────────────────────────────
# Synthetic failure mode templates for augmentation
# ─────────────────────────────────────────────────────────────

FAILURE_MODES = [
    {
        "failure_mode": "PostgreSQL connection pool exhaustion",
        "services": ["web-api", "postgres", "redis-cache"],
        "scale": "50k RPM, 500k users",
        "trigger": "Traffic spike during marketing campaign",
    },
    {
        "failure_mode": "Redis cache stampede after restart",
        "services": ["web-api", "redis-primary", "postgres"],
        "scale": "20k RPM, e-commerce platform",
        "trigger": "Redis OOM restart at 3AM",
    },
    {
        "failure_mode": "Memory leak in Node.js service introduced in v3.2.1",
        "services": ["checkout-svc", "orders-db", "payment-gateway"],
        "scale": "10k RPM checkout flow",
        "trigger": "Deployment at 14:30 UTC",
    },
    {
        "failure_mode": "Kafka consumer group lag causing order processing backlog",
        "services": ["order-processor", "kafka", "fulfillment-svc", "postgres"],
        "scale": "100k orders/day",
        "trigger": "Consumer crash loop due to malformed message",
    },
    {
        "failure_mode": "DNS resolution failure for external payment provider",
        "services": ["checkout-svc", "payment-gateway (external)", "orders-db"],
        "scale": "Global e-commerce platform",
        "trigger": "DNS TTL expiry during traffic peak",
    },
    {
        "failure_mode": "Kubernetes pod OOMKilled due to missing memory limit",
        "services": ["analytics-svc", "clickhouse", "kafka"],
        "scale": "Data pipeline processing 1M events/hour",
        "trigger": "Large backfill query consuming unbounded memory",
    },
    {
        "failure_mode": "TLS certificate expiry on API gateway",
        "services": ["nginx-gateway", "web-api", "auth-svc"],
        "scale": "SaaS platform, 50k active users",
        "trigger": "Certificate expired at midnight UTC",
    },
    {
        "failure_mode": "Cascading timeout: slow Postgres query → connection pool exhaustion → upstream 503s",
        "services": ["web-api", "postgres-primary", "nginx", "redis"],
        "scale": "Enterprise SaaS, 10k concurrent users",
        "trigger": "Missing index on a foreign key added in last migration",
    },
]

INCIDENT_TIMES = [
    "02:34 UTC (3AM weekend)",
    "14:22 UTC (peak business hours)",
    "09:15 UTC (Monday morning)",
    "23:58 UTC (end of day deployment)",
    "16:45 UTC (Friday afternoon)",
]


async def call_vllm(client: aiohttp.ClientSession, base_url: str, messages: list, model: str = "Qwen/Qwen2.5-72B-Instruct", api_key: str = "synthesis") -> str:
    """Call a vLLM OpenAI-compatible endpoint."""
    resp = await client.post(
        f"{base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": messages, "max_tokens": 4096, "temperature": 0.8},
        timeout=aiohttp.ClientTimeout(total=120.0),
    )
    resp.raise_for_status()
    return (await resp.json())["choices"][0]["message"]["content"]


async def call_claude(client: aiohttp.ClientSession, messages: list, api_key: str, system_prompt: str) -> str:
    """Call the Anthropic Claude API as a fallback."""
    resp = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": messages,
        },
        timeout=aiohttp.ClientTimeout(total=120.0),
    )
    resp.raise_for_status()
    return (await resp.json())["content"][0]["text"]


class BulkSynthesizer:
    def __init__(
        self,
        vllm_endpoints: list[str] | None = None,
        vllm_endpoint_1: str | None = None,
        vllm_endpoint_2: str | None = None,
        model_name: str = "Qwen/Qwen2.5-72B-Instruct",
        anthropic_api_key: str | None = None,
    ):
        # Support both old 2-endpoint API and new 4-endpoint VLLM_URLS list
        if vllm_endpoints:
            self.endpoints = vllm_endpoints
        else:
            urls_env = os.environ.get("VLLM_URLS", "")
            if urls_env:
                self.endpoints = [u.strip() for u in urls_env.split(",") if u.strip()]
            else:
                self.endpoints = [ep for ep in [vllm_endpoint_1, vllm_endpoint_2] if ep]

        self.model_name = model_name
        self.anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.vllm_api_key = os.environ.get("VLLM_API_KEY", "synthesis")
        self._endpoint_idx = 0
        self._lock = asyncio.Lock()

    async def _next_endpoint(self) -> str | None:
        """Round-robin across available endpoints (async-safe)."""
        if not self.endpoints:
            return None
        async with self._lock:
            idx = self._endpoint_idx % len(self.endpoints)
            self._endpoint_idx += 1
        return self.endpoints[idx]

    async def _call_vllm(
        self,
        session: aiohttp.ClientSession,
        system: str,
        user: str,
        max_tokens: int = 2048,
    ) -> str | None:
        """Call vLLM OpenAI-compatible endpoint with round-robin load balancing."""
        endpoint = await self._next_endpoint()
        if not endpoint:
            return None
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return await call_vllm(session, endpoint, messages, self.model_name, self.vllm_api_key)
        except Exception as e:
            logger.debug(f"vLLM call failed at {endpoint}: {e}")
            return None

    async def _call_anthropic_async(
        self,
        session: aiohttp.ClientSession,
        system: str,
        user: str,
    ) -> str | None:
        """Async Anthropic API fallback."""
        if not self.anthropic_key:
            return None
        try:
            messages = [{"role": "user", "content": user}]
            return await call_claude(session, messages, self.anthropic_key, system)
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return None

    def _call_anthropic(self, system: str, user: str, max_tokens: int = 2048) -> str | None:
        """Synchronous Anthropic API fallback (legacy)."""
        if not HAS_ANTHROPIC or not self.anthropic_key:
            return None
        try:
            client = anthropic.Anthropic(api_key=self.anthropic_key)
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return None

    def _parse_json(self, text: str | None) -> dict | None:
        """Extract JSON from model response."""
        if not text:
            return None
        import re
        # Try direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        # Try extracting JSON block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None

    async def synthesize_sft_pairs(
        self,
        raw_docs: list[dict],
        output_path: Path,
        batch_size: int = 20,
    ) -> int:
        """Synthesize SFT training pairs from raw documents."""
        count = 0
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(raw_docs), batch_size):
                batch = raw_docs[i:i+batch_size]
                tasks = [
                    self._synthesize_sft_single(session, doc)
                    for doc in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                with open(output_path, "a") as f:
                    for result in results:
                        if isinstance(result, dict):
                            f.write(json.dumps(result) + "\n")
                            count += 1

                logger.info(f"SFT synthesis: {i+len(batch)}/{len(raw_docs)} ({count} pairs generated)")

        return count

    async def _synthesize_sft_single(
        self, session: aiohttp.ClientSession, doc: dict
    ) -> dict | None:
        """Synthesize one SFT pair from a raw document."""
        content = doc.get("raw_text", doc.get("content", ""))
        if len(content) < 300:
            return None

        user = SFT_EXTRACTION_USER_TEMPLATE.format(
            source=doc.get("source_url", "unknown"),
            content=content[:8000],
        )

        text = await self._call_vllm(session, SFT_EXTRACTION_SYSTEM, user)
        if not text:
            text = self._call_anthropic(SFT_EXTRACTION_SYSTEM, user)

        result = self._parse_json(text)
        if result:
            result["_source"] = doc.get("source_url", "")
        return result

    async def synthesize_from_templates(
        self,
        output_path: Path,
        n_per_template: int = 3,
    ) -> int:
        """Synthesize incidents from failure mode templates."""
        count = 0
        async with aiohttp.ClientSession() as session:
            tasks = []
            for template in FAILURE_MODES:
                for _ in range(n_per_template):
                    incident_time = random.choice(INCIDENT_TIMES)
                    tasks.append(self._synthesize_from_template(session, template, incident_time))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            with open(output_path, "a") as f:
                for result in results:
                    if isinstance(result, dict):
                        f.write(json.dumps(result) + "\n")
                        count += 1

        logger.info(f"Template synthesis: {count} incidents generated")
        return count

    async def _synthesize_from_template(
        self, session: aiohttp.ClientSession, template: dict, incident_time: str
    ) -> dict | None:
        user = INCIDENT_SYNTHESIS_USER_TEMPLATE.format(
            incident_time=incident_time,
            **template,
        )
        text = await self._call_vllm(session, INCIDENT_SYNTHESIS_SYSTEM, user)
        if not text:
            text = self._call_anthropic(INCIDENT_SYNTHESIS_SYSTEM, user)
        return self._parse_json(text)

    async def synthesize_dpo_pairs(
        self,
        sft_pairs: list[dict],
        output_path: Path,
    ) -> int:
        """Synthesize DPO preference pairs from SFT examples."""
        count = 0
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._synthesize_dpo_single(session, pair)
                for pair in sft_pairs
                if pair.get("postmortem_draft", {}).get("root_cause")
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            with open(output_path, "a") as f:
                for result in results:
                    if isinstance(result, dict):
                        f.write(json.dumps(result) + "\n")
                        count += 1

        logger.info(f"DPO synthesis: {count} preference pairs generated")
        return count

    async def _synthesize_dpo_single(
        self, session: aiohttp.ClientSession, pair: dict
    ) -> dict | None:
        root_cause = pair.get("postmortem_draft", {}).get("root_cause", "")
        incident_json = json.dumps({
            "alerts": pair.get("alerts", []),
            "context": pair.get("context", {}),
            "logs": pair.get("logs", ""),
        }, indent=2)

        user = DPO_USER_TEMPLATE.format(
            incident_json=incident_json[:3000],
            root_cause=root_cause,
        )
        text = await self._call_vllm(session, DPO_SYSTEM, user)
        if not text:
            text = self._call_anthropic(DPO_SYSTEM, user)
        return self._parse_json(text)


def main():
    parser = argparse.ArgumentParser(description="OncallCompass bulk synthesis")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--vllm_endpoint_1", default=os.environ.get("VLLM_ENDPOINT_1"))
    parser.add_argument("--vllm_endpoint_2", default=os.environ.get("VLLM_ENDPOINT_2"))
    parser.add_argument("--vllm_urls", default=os.environ.get("VLLM_URLS", ""),
                        help="Comma-separated vLLM endpoints, e.g. http://localhost:8001,http://localhost:8002")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--n_per_template", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse endpoint list (VLLM_URLS takes priority over individual endpoint args)
    vllm_endpoints = None
    if args.vllm_urls:
        vllm_endpoints = [u.strip() for u in args.vllm_urls.split(",") if u.strip()]

    synthesizer = BulkSynthesizer(
        vllm_endpoints=vllm_endpoints,
        vllm_endpoint_1=args.vllm_endpoint_1,
        vllm_endpoint_2=args.vllm_endpoint_2,
        model_name=args.model_name,
    )

    # Load raw documents
    raw_docs = []
    for jsonl_file in Path(args.raw_dir).glob("**/*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        raw_docs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    logger.info(f"Loaded {len(raw_docs)} raw documents")

    sft_path = output_dir / "sft_pairs.jsonl"

    # SFT from raw docs
    sft_from_raw = asyncio.run(synthesizer.synthesize_sft_pairs(raw_docs, sft_path))

    # SFT from templates (augmentation)
    sft_from_templates = asyncio.run(synthesizer.synthesize_from_templates(sft_path, args.n_per_template))

    # Load all SFT pairs for DPO
    sft_pairs = []
    with open(sft_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sft_pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # DPO pairs
    dpo_path = output_dir / "dpo_pairs.jsonl"
    dpo_count = asyncio.run(synthesizer.synthesize_dpo_pairs(
        random.sample(sft_pairs, min(len(sft_pairs), max(1, len(sft_pairs) // 3))),
        dpo_path,
    ))

    logger.info(f"Synthesis complete: {sft_from_raw + sft_from_templates} SFT, {dpo_count} DPO")


if __name__ == "__main__":
    main()
