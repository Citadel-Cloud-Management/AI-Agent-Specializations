"""CodingAgent - Writes, debugs, reviews, and explains code."""
from __future__ import annotations
import json
from typing import Any
from agents.shared.base_agent import AgentConfig, AgentResult, BaseAgent
from agents.shared.guardrails import check_injection
from agents.shared.llm_client import LLMRequest, MultiCloudLLM

CFG = AgentConfig("coding-agent", "Writes, debugs, reviews, and explains code")

class CodingAgent(BaseAgent):
    def __init__(self, config: AgentConfig = CFG, llm: MultiCloudLLM | None = None, **kw: Any) -> None:
        super().__init__(config)
        self._llm = llm or MultiCloudLLM()

    def validate_input(self, d: dict) -> bool:
        return bool(d.get("prompt"))

    async def execute(self, d: dict) -> AgentResult:
        if not self.validate_input(d):
            return self._err("Need 'prompt'")
        if not check_injection(str(d.get("prompt", ""))).passed:
            return self._err("Blocked by guardrails")
        self._log(f"Executing coding-agent")
        resp = await self._llm.complete(LLMRequest(
            str(d["prompt"]), system="You are a coding agent. Generate clean tested code. Return JSON.", temperature=0.2,
        ))
        try:
            result = json.loads(resp.content)
        except json.JSONDecodeError:
            result = {"output": resp.content}
        return self._ok(result)
