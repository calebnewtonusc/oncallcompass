"""
OncallCompass Triage Agent.

Main inference interface for incident triage. Given alert signals and stack
context, the agent:
    1. Ranks root cause hypotheses by likelihood
    2. Sequences investigation steps
    3. Optionally generates a postmortem draft

The agent wraps the trained OncallCompass model and provides a clean
Python API plus an optional FastAPI HTTP server.

Usage — Python API:
    from agents.triage_agent import TriageAgent

    agent = TriageAgent(model_path="checkpoints/rl")
    result = agent.triage(
        alerts=["5xx spike on /api/checkout", "latency p99 +340ms"],
        context={"last_deploy": "6h ago", "stack": ["nginx", "node", "postgres", "redis"]}
    )
    print(result.ranked_hypotheses[0])

Usage — HTTP server:
    uvicorn agents.triage_agent:app --host 0.0.0.0 --port 8000

    POST /triage
    {
      "alerts": ["..."],
      "logs": "...",
      "metrics": {},
      "context": {}
    }
"""

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# Optional FastAPI import — only needed for HTTP server mode
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


SYSTEM_PROMPT = """You are OncallCompass, a specialized incident response AI trained on 300k incident-to-resolution traces.

Given alert signals and stack context for an active incident, you:
1. Rank root cause hypotheses by likelihood — the most probable cause always comes first
2. Sequence investigation steps in the order a senior SRE would follow them
3. Generate structured, prevention-focused postmortem drafts when requested

Respond ONLY with valid JSON matching the OncallCompass output schema."""

OUTPUT_SCHEMA = {
    "ranked_hypotheses": [
        {
            "hypothesis": "string",
            "confidence": "float 0-1",
            "evidence": ["string"],
            "ruling_out": "string — single step to confirm or rule out",
        }
    ],
    "investigation_steps": ["ordered list of investigation steps"],
    "postmortem_draft": {
        "summary": "string",
        "timeline": ["string"],
        "root_cause": "string",
        "contributing_factors": ["string"],
        "action_items": [{"item": "string", "owner": "string", "prevents": "string"}],
    },
}


@dataclass
class Hypothesis:
    hypothesis: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    ruling_out: str = ""


@dataclass
class PostmortemDraft:
    summary: str = ""
    timeline: list[str] = field(default_factory=list)
    root_cause: str = ""
    contributing_factors: list[str] = field(default_factory=list)
    action_items: list[dict[str, str]] = field(default_factory=list)


@dataclass
class TriageResult:
    ranked_hypotheses: list[Hypothesis]
    investigation_steps: list[str]
    postmortem_draft: PostmortemDraft | None
    latency_ms: float
    model_path: str

    def top_hypothesis(self) -> Hypothesis | None:
        return self.ranked_hypotheses[0] if self.ranked_hypotheses else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranked_hypotheses": [asdict(h) for h in self.ranked_hypotheses],
            "investigation_steps": self.investigation_steps,
            "postmortem_draft": asdict(self.postmortem_draft)
            if self.postmortem_draft
            else None,
            "meta": {
                "latency_ms": self.latency_ms,
                "model_path": self.model_path,
                "hypothesis_count": len(self.ranked_hypotheses),
                "step_count": len(self.investigation_steps),
            },
        }


class TriageAgent:
    """OncallCompass triage agent — wraps the trained model for inference."""

    def __init__(
        self,
        model_path: str = "checkpoints/rl",
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        device: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading OncallCompass from {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        # Set a distinct pad token so that padding does not collide with the EOS
        # token, which would cause the model to stop generating at padded positions.
        if self.tokenizer.pad_token_id is None or (
            self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        ):
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device.startswith("cuda") else None,
            trust_remote_code=True,
        )
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        print("OncallCompass loaded.")

    def triage(
        self,
        alerts: list[str],
        logs: str = "",
        metrics: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        include_postmortem: bool = False,
    ) -> TriageResult:
        """
        Triage an incident and return ranked hypotheses + investigation steps.

        Args:
            alerts: List of alert strings currently firing
            logs: Optional log snippet from the incident
            metrics: Optional dict of current metric values
            context: Optional dict with keys: last_deploy, stack, recent_changes
            include_postmortem: Whether to generate a postmortem draft

        Returns:
            TriageResult with ranked_hypotheses and investigation_steps
        """
        prompt = self._build_prompt(alerts, logs, metrics or {}, context or {})

        start = time.time()
        response_text = self._generate(prompt)
        latency_ms = (time.time() - start) * 1000

        response_dict = self._parse_response(response_text)
        return self._build_result(response_dict, latency_ms, include_postmortem)

    def _build_prompt(
        self,
        alerts: list[str],
        logs: str,
        metrics: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        user_content = json.dumps(
            {
                "alerts": alerts,
                "logs": logs,
                "metrics": metrics,
                "context": context,
            },
            indent=2,
        )

        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}

    def _build_result(
        self,
        response: dict[str, Any],
        latency_ms: float,
        include_postmortem: bool,
    ) -> TriageResult:
        hypotheses = [
            Hypothesis(
                hypothesis=h.get("hypothesis", ""),
                confidence=float(h.get("confidence", 0.0)),
                evidence=h.get("evidence", []),
                ruling_out=h.get("ruling_out", ""),
            )
            for h in response.get("ranked_hypotheses", [])
        ]

        steps = response.get("investigation_steps", [])

        postmortem = None
        if include_postmortem and "postmortem_draft" in response:
            pd = response["postmortem_draft"]
            postmortem = PostmortemDraft(
                summary=pd.get("summary", ""),
                timeline=pd.get("timeline", []),
                root_cause=pd.get("root_cause", ""),
                contributing_factors=pd.get("contributing_factors", []),
                action_items=pd.get("action_items", []),
            )

        return TriageResult(
            ranked_hypotheses=hypotheses,
            investigation_steps=steps,
            postmortem_draft=postmortem,
            latency_ms=latency_ms,
            model_path=self.model_path,
        )


# ─── FastAPI HTTP server ─────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="OncallCompass",
        description="ML-powered incident triage — ranked hypotheses, investigation steps, MTTR reduction",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _agent: TriageAgent | None = None

    def get_agent() -> TriageAgent:
        global _agent
        if _agent is None:
            model_path = os.environ.get("FINAL_CHECKPOINT", "checkpoints/rl")
            _agent = TriageAgent(model_path=model_path)
        return _agent

    class TriageRequest(BaseModel):
        alerts: list[str]
        logs: str = ""
        metrics: dict[str, Any] = {}
        context: dict[str, Any] = {}
        include_postmortem: bool = False

    class TriageResponse(BaseModel):
        ranked_hypotheses: list[dict[str, Any]]
        investigation_steps: list[str]
        postmortem_draft: dict[str, Any] | None
        meta: dict[str, Any]

    @app.post("/triage", response_model=TriageResponse)
    async def triage_endpoint(request: TriageRequest) -> dict[str, Any]:
        """Triage an active incident."""
        if not request.alerts:
            raise HTTPException(
                status_code=400, detail="At least one alert is required"
            )

        agent = get_agent()
        result = agent.triage(
            alerts=request.alerts,
            logs=request.logs,
            metrics=request.metrics,
            context=request.context,
            include_postmortem=request.include_postmortem,
        )
        return result.to_dict()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "model": os.environ.get("FINAL_CHECKPOINT", "checkpoints/rl"),
        }


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OncallCompass triage agent")
    parser.add_argument("--model_path", default="checkpoints/rl")
    parser.add_argument("--alerts", nargs="+", required=True)
    parser.add_argument("--logs", default="")
    parser.add_argument("--stack", nargs="+", default=[])
    parser.add_argument("--last_deploy", default="")
    parser.add_argument("--postmortem", action="store_true")
    args = parser.parse_args()

    agent = TriageAgent(model_path=args.model_path)
    result = agent.triage(
        alerts=args.alerts,
        logs=args.logs,
        context={
            "stack": args.stack,
            "last_deploy": args.last_deploy,
        },
        include_postmortem=args.postmortem,
    )

    print(json.dumps(result.to_dict(), indent=2))
