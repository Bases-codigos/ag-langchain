from app.config import settings
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.models.factory import ModelFactory
from app.prompts.manager import PromptManager
from app.routing.agent_router import AgentRouter
from app.schemas import UserMessage
from app.tools.registry import ToolRegistry
from app.agent.service import AgentService


def build_agent_service() -> AgentService:
    return AgentService(
        router=AgentRouter(),
        model_factory=ModelFactory(),
        prompt_manager=PromptManager(),
        tool_registry=ToolRegistry(),
        short_memory=ShortTermMemory(max_messages=settings.max_short_term_messages),
        long_memory=LongTermMemory(),
    )


if __name__ == "__main__":
    service = build_agent_service()

    messages = [
        UserMessage(
            user_id="harold",
            thread_id="thread-1",
            message="Recuerda que prefiero respuestas cortas y técnicas",
        ),
        UserMessage(
            user_id="harold",
            thread_id="thread-1",
            message="Explícame qué es LangGraph",
        ),
        UserMessage(
            user_id="harold",
            thread_id="thread-1",
            message="Cuánto es 25 * 4 + 10",
        ),
    ]

    for msg in messages:
        response = service.invoke(msg)
        print("\n--- RESPUESTA ---")
        print(response.model_dump())