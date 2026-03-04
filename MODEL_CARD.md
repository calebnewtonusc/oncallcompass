# OncallCompass — Model Card

## Model Details

| Field | Value |
|---|---|
| Model name | OncallCompass v1.0 |
| Base model | Qwen/Qwen2.5-7B-Coder-Instruct |
| Parameters | 7.6B |
| Architecture | Transformer decoder (Qwen2.5) |
| Fine-tuning method | SFT → GRPO RL → DPO |
| Training hardware | 18× NVIDIA A6000 48GB |
| Training data | ~50k incident-to-resolution examples |
| Developer | calebnewtonusc |
| License | MIT |

---

## Intended Use

OncallCompass is designed for:
- Production incident triage: clustering alert storms into incident hypotheses
- Root cause ranking: ordering hypotheses by causal probability
- Investigation sequencing: ordering investigation steps as a senior SRE would
- Postmortem generation: structured 5-Why analysis with action items

**NOT designed for:**
- Security incident response (different threat model)
- Legal or compliance decisions
- Autonomous system remediation without human approval

---

## Training Data

See [DATA_SOURCES.md](DATA_SOURCES.md) for full breakdown.

- 40% public postmortems (GitHub, engineering blogs)
- 25% Stack Overflow / ServerFault incident Q&A
- 20% AWS/GCP/Azure status page history
- 15% synthetic scenarios from dependency graph generator

**Data filters**: Only postmortems with explicit root cause, ≥3 action items, named service stack.

---

## Evaluation

### CompassBench Results (v1.0 target)

| Metric | Target | Description |
|---|---|---|
| Top-1 root cause accuracy | 75%+ | Correct RC in position 1 |
| Top-3 root cause accuracy | 92%+ | Correct RC in top 3 |
| MTTR reduction vs. baseline | 25%+ | vs. median junior SRE time |
| Investigation step efficiency | 85%+ | Steps that move toward RC |
| Postmortem action item quality | 80%+ | Actionable, not vague |
| 5-Why completeness | 90%+ | All 5 levels populated |

### CompassBench Categories

| Category | Scenarios |
|---|---|
| Database incidents | 30 |
| Network partitions | 25 |
| Memory pressure / OOM | 25 |
| Deployment regressions | 30 |
| Cascading failures | 25 |
| External dependency | 15 |

---

## Limitations

- **Training distribution**: Primarily trained on public postmortems from tech companies. May underperform on unusual/proprietary infrastructure.
- **Context window**: Limited to 8192 tokens — very large log dumps may need truncation.
- **JSON output**: Structured output relies on format adherence; malformed JSON can occur on unusual inputs.
- **Confidence calibration**: Hypothesis confidence scores are relative, not calibrated probabilities.
- **No live access**: The model does not query live systems — it reasons from provided context only.

---

## Ethical Considerations

- OncallCompass is an assistive tool, not an autonomous decision-maker
- Human SREs must review all root cause conclusions before acting
- Training data includes publicly disclosed incidents only
- No personally identifiable information was included in training data
