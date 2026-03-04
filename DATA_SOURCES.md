# OncallCompass — Data Sources

OncallCompass is trained on ~50k incident-to-resolution examples across 4 primary streams.

---

## Stream 1: Public Postmortem Corpus (40%)

~20k documents from public SRE repositories and engineering blogs.

**GitHub repositories:**
- `danluu/post-mortems` — 300+ curated postmortems
- `upgundecha/howtheysre` — SRE practices across 30+ companies
- Individual company SRE repos (Google, Netflix, Cloudflare, Stripe)

**Engineering blogs:**
- Google SRE Blog (`sre.google/sre-book/`)
- Atlassian Incident Handbook
- PagerDuty Incident Response Guide
- Netflix Tech Blog (incident/reliability posts)
- Stripe Engineering Blog
- Cloudflare Blog (outage posts)
- Discord Engineering (availability posts)

**Format after processing:**
```json
{
  "alerts": ["derived from timeline"],
  "context": {"stack": [...], "last_deploy": "..."},
  "ranked_hypotheses": [...],
  "investigation_steps": [...],
  "postmortem_draft": {
    "root_cause": "...",
    "contributing_factors": [...],
    "five_why": [...],
    "action_items": [...]
  }
}
```

**Quality filters:**
- Only postmortems with explicit root cause statement
- Only postmortems with at least 3 action items
- Exclude postmortems with vague root causes ("unknown" / "monitoring gap")

---

## Stream 2: Stack Overflow + Server Fault (25%)

~12.5k Q&A pairs about production incidents.

**Source**: Stack Exchange Data Dump
- `stackoverflow.com` — tagged: `[incident-response]`, `[pagerduty]`, `[on-call]`, `[sre]`
- `serverfault.com` — production incidents, infrastructure failures
- `devops.stackexchange.com` — SRE and DevOps questions

**Filter criteria:**
- Score ≥ 5 (community-validated quality)
- Has accepted answer
- Question describes a concrete incident pattern
- Answer provides actionable investigation steps

**Synthesis**: Each Q&A is converted to an (alert context, investigation steps, resolution) triple.

---

## Stream 3: Cloud Provider Status Pages (20%)

~10k incidents from AWS, GCP, Azure, and Cloudflare.

**Sources:**
- AWS Health Dashboard (`health.aws.amazon.com`) — full incident history
- Google Cloud Status (`status.cloud.google.com`) — incident details
- Azure Status (`status.azure.com`) — service disruption history
- Cloudflare System Status — network incidents

**Per-incident extraction:**
- Start time, affected services, symptoms
- Investigation progression (timeline entries)
- Root cause (from final resolution)
- Mitigation steps taken

**Value**: Real-world cloud infrastructure failure patterns with confirmed root causes.

---

## Stream 4: Synthetic Incident Generation (15%)

~7.5k scenarios generated from service dependency graph templates.

**Generator approach:**
1. Define service topology (microservices with dependency edges)
2. Inject failure modes (CPU spike, memory leak, connection pool exhaustion, network partition)
3. Generate realistic alert storm from failure propagation model
4. Synthesize investigation path using domain rules
5. Validate with `claude-opus-4-6` for realism

**Failure mode coverage:**
- Database: connection pool exhaustion, slow queries, replication lag
- Cache: cache stampede, memory eviction, TTL misconfiguration
- Network: DNS resolution failure, TLS certificate expiry, partition
- Memory: OOM, memory leak, GC pressure
- Deployment: bad config push, version incompatibility, rollout stuck
- External: third-party API timeout, CDN cache poisoning

---

## Data Quality Standards

| Criterion | Threshold |
|---|---|
| Root cause explicitness | Must have explicit "root cause" statement |
| Action items | ≥ 3 concrete action items |
| Timeline completeness | Must have start time and resolution time |
| Stack context | Must name at least 2 specific services |
| Duplicate removal | SHA-256 dedup on root cause text |

---

## License and Attribution

All public postmortem content is attributed to original authors and used under
applicable Creative Commons licenses. Stack Overflow content is licensed under
CC BY-SA 4.0. Cloud provider status pages are public information.
