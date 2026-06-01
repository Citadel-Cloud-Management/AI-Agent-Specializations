"""CrossCloudOrchestrator - Framework-level orchestration across any cloud provider."""
from __future__ import annotations
import json
from typing import Any
from agents.shared.base_agent import AgentConfig, AgentResult, BaseAgent
from agents.shared.guardrails import check_injection
from agents.shared.llm_client import LLMRequest, MultiCloudLLM

CFG = AgentConfig("cross-cloud-orchestrator", "Framework-level orchestration across any cloud provider")

class CrossCloudOrchestrator(BaseAgent):
    def __init__(self, config: AgentConfig = CFG, llm: MultiCloudLLM | None = None, **kw: Any) -> None:
        super().__init__(config)
        self._llm = llm or MultiCloudLLM()

    def validate_input(self, d: dict) -> bool:
        return bool(d.get("action"))

    async def execute(self, d: dict) -> AgentResult:
        if not self.validate_input(d):
            return self._err("Need 'action'")
        if not check_injection(str(d.get("action", ""))).passed:
            return self._err("Blocked by guardrails")
        self._log(f"Executing cross-cloud-orchestrator")
        resp = await self._llm.complete(LLMRequest(
            str(d["action"]), system="You orchestrate across clouds. Route, failover, govern data. Return JSON.", temperature=0.2,
        ))
        try:
            result = json.loads(resp.content)
        except json.JSONDecodeError:
            result = {"output": resp.content}
        return self._ok(result)
