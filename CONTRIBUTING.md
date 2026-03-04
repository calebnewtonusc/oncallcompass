# Contributing to OncallCompass

## Ways to Contribute

1. **Postmortem submissions** — Add public postmortems to the training corpus
2. **CompassBench scenarios** — Submit new incident scenarios for evaluation
3. **Bug reports** — File issues for incorrect root cause rankings
4. **Training improvements** — Propose reward function changes or new data sources

## Development Setup

```bash
git clone https://github.com/calebnewtonusc/oncallcompass
cd oncallcompass
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Adding Postmortems

Postmortems must meet quality standards:

1. Explicit root cause statement (not "unknown")
2. At least 3 concrete action items
3. Timeline with start and resolution times
4. Named specific services (not "backend service")

Format as JSONL:
```json
{
  "source": "https://example.com/postmortem",
  "alerts": ["High error rate on /api/checkout", "P99 latency +400ms"],
  "context": {
    "stack": ["nginx", "node", "postgres", "redis"],
    "last_deploy": "2h before incident"
  },
  "root_cause": "Redis connection pool exhaustion caused by connection leak in v2.3.1",
  "investigation_steps": [...],
  "action_items": [...]
}
```

## Adding CompassBench Scenarios

New scenarios belong in `data/drills/` as JSONL entries:

```json
{
  "id": "cb_xxx",
  "category": "database|network|memory|deployment|cascade|external",
  "difficulty": "easy|medium|hard",
  "alerts": [...],
  "context": {...},
  "ground_truth": {
    "root_cause": "...",
    "expert_steps": [...],
    "baseline_steps": 8
  }
}
```

## Code Style

- Python 3.11+
- `ruff` for linting
- `black` for formatting
- Type annotations required on all public functions
- `loguru` for logging (not `print`)

```bash
ruff check .
black .
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Add tests for new evaluation scenarios
4. Ensure `scripts/check_env.sh` passes
5. Submit PR with description of changes

## Reporting Incorrect Root Cause Rankings

If OncallCompass ranks a root cause incorrectly on a real incident:

1. Open an issue with the incident context (anonymized)
2. Include the model's output and the actual root cause
3. Label the issue `root-cause-error`

These cases are the most valuable training signal for the next version.
