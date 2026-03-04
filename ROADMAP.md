# OncallCompass — Roadmap

## v1.0 — Foundation (Current)

**Goal**: Single-incident root cause ranking with measured MTTR reduction.

- [x] Architecture design and data pipeline
- [x] SFT training on 50k incident traces
- [x] GRPO/PPO RL with MTTR reward
- [x] DPO alignment on fast/slow path pairs
- [x] CompassBench: 150 incident scenarios
- [ ] Discovery: postmortem corpus (40k documents)
- [ ] Discovery: AWS/GCP/Azure status pages
- [ ] Synthesis: incident_synthesizer.py
- [ ] Triage agent: alert clustering
- [ ] Hypothesis agent: causal probability scoring
- [ ] Investigation agent: runbook execution
- [ ] Postmortem agent: 5-Why generation
- [ ] Target: 75% top-1 root cause accuracy on CompassBench

---

## v1.5 — Service Topology

**Goal**: Incorporate service dependency graph for topology-aware hypothesis ranking.

- [ ] Graph-based alert correlation (service mesh topology)
- [ ] Causal graph reasoning (upstream/downstream failure propagation)
- [ ] Integration with: Datadog, PagerDuty, OpsGenie
- [ ] Slack bot for real-time incident response
- [ ] Target: 85% top-1 root cause accuracy

---

## v2.0 — Live Runbook Execution

**Goal**: Active investigation agent that queries systems during incidents.

- [ ] Live log querying (CloudWatch, Datadog Logs, Loki)
- [ ] Live metrics queries (Prometheus, Datadog Metrics)
- [ ] Kubernetes event stream correlation
- [ ] Deployment diff integration (GitHub, ArgoCD)
- [ ] Automated runbook execution with approval gates
- [ ] Target: 30% MTTR reduction vs. unassisted SRE

---

## v3.0 — Proactive Detection

**Goal**: Detect incidents before they page.

- [ ] Anomaly detection pre-training layer
- [ ] Drift detection from deployment diffs
- [ ] Capacity exhaustion prediction
- [ ] Pre-incident runbook generation
- [ ] Target: 20% of incidents resolved before paging on-call
