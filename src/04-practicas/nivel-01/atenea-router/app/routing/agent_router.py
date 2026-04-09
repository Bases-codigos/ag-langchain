from app.config import settings
from app.schemas import RoutingDecision

class AgentRouter:
    """
    Decide dinámicamente:
    - modo
    - modelo
    - prompt
    - herramientas
    """
    def route(self, messages: str) -> RoutingDecision:
        lowered = messages.lower()

        if "recuerda que" in lowered or "a partir de ahora" in lowered:
            return RoutingDecision(
                mode="memory_update",
                model=settings.default_model,
                system_prompt_key="memory_manager",
                tool_names=[],
            )

        if any(word in lowered for word in ["arquitectura", "error", "python", "langchain", "langgraph", "api"]):
            return RoutingDecision(
                mode="technical",
                model_name=settings.reasoning_model,
                system_prompt_key="technical_assistant",
                tool_names=["kb_search", "user_profile"],
            )

        if any(symbol in lowered for symbol in ["+", "-", "*", "/"]):
            return RoutingDecision(
                mode="general",
                model_name=settings.default_model,
                system_prompt_key="general_assistant",
                tool_names=["calculator"],
            )
        return RoutingDecision(
            mode="general",
            model_name=settings.default_model,
            system_prompt_key="general_assistant",
            tool_names=["user_profile"],
        )