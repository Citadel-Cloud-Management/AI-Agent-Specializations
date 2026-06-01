"""Citadel Agent Orchestrator - Unified entry point for all 15 agents."""
import asyncio
import importlib
from typing import Any
from agents.shared.base_agent import AgentResult, BaseAgent
from agents.shared.llm_client import MultiCloudLLM

AGENT_MAP = {
    "research": ("agents.01-research-agent", "ResearchAgent"),
    "recommendation": ("agents.02-recommendation-agent", "RecommendationAgent"),
    "customer-support": ("agents.03-customer-support-agent", "CustomerSupportAgent"),
    "coding": ("agents.04-coding-agent", "CodingAgent"),
    "browser": ("agents.05-browser-agent", "BrowserAgent"),
    "multi-research-team": ("agents.06-multi-agent-research-team", "MultiAgentResearchTeam"),
    "data-analyst": ("agents.07-data-analyst-agent", "DataAnalystAgent"),
    "conversational-copilot": ("agents.08-conversational-copilot", "ConversationalCopilot"),
    "autonomous-task": ("agents.09-autonomous-task-agent", "AutonomousTaskAgent"),
    "devops": ("agents.10-devops-agent", "DevOpsAgent"),
    "data-analytics": ("agents.11-data-analytics-agent", "DataAnalyticsAgent"),
    "security-governance": ("agents.12-security-governance-agent", "SecurityGovernanceAgent"),
    "industry": ("agents.13-industry-agent", "IndustryAgent"),
    "infrastructure": ("agents.14-infrastructure-agent", "InfrastructureAgent"),
    "cross-cloud": ("agents.15-cross-cloud-orchestrator", "CrossCloudOrchestrator"),
}

class Orchestrator:
    def __init__(self, llm: MultiCloudLLM | None = None) -> None:
        self._llm = llm or MultiCloudLLM()
        self._cache: dict[str, BaseAgent] = {}

    def _get(self, name: str) -> BaseAgent:
        if name not in self._cache:
            entry = AGENT_MAP.get(name)
            if not entry:
                raise ValueError(f"Unknown agent: {name}. Available: {sorted(AGENT_MAP)}")
            mod, cls_name = entry
            m = importlib.import_module(mod)
            cls = getattr(m, cls_name)
            self._cache[name] = cls(llm=self._llm)
        return self._cache[name]

    async def run(self, agent: str, data: dict) -> AgentResult:
        try:
            return await self._get(agent).execute(data)
        except Exception as e:
            return AgentResult("orchestrator", "error", error=str(e))

    async def run_parallel(self, reqs: list[dict[str, Any]]) -> list[AgentResult]:
        return await asyncio.gather(*(self.run(r["agent"], r.get("input", {})) for r in reqs))

    def list_agents(self) -> list[dict[str, str]]:
        out = []
        for key, (mod, cls_name) in sorted(AGENT_MAP.items()):
            try:
                m = importlib.import_module(mod)
                inst = getattr(m, cls_name)()
                out.append({"name": key, "desc": inst.config.description, "ver": inst.config.version})
            except Exception:
                out.append({"name": key, "desc": "(err)", "ver": "?"})
        return out
